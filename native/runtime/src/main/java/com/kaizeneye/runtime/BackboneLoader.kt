package com.kaizeneye.runtime

import android.content.Context
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.delay
import java.io.File

/**
 * Runtime.loadBackbone: locate variants -> decision cache -> (benchmark in ":probe") -> create the chosen candidate in the
 * app process under the crash-guard marker -> warm it up.
 */
internal object BackboneLoader {
    const val PROBE_TIMEOUT_MS = 60_000L

    /** 60 s per candidate, scaled up for big models (first NPU JIT compile of an 88 MB DINOv2 takes longer than R18's). */
    fun probeTimeoutMs(modelBytes: Long): Long =
        maxOf(PROBE_TIMEOUT_MS, (PROBE_TIMEOUT_MS * (modelBytes.toDouble() / (30L shl 20))).toLong())

    private class Step(val measurement: Measurement, val output: FloatArray?, val probeProblem: String?)

    suspend fun load(ctx: Context, spec: BackboneSpec, forceAccel: Accel?, rebenchmark: Boolean): FeatureBackbone {
        NpuEnv.ensure(ctx)
        validate(spec)
        val store = DecisionStore.get(ctx)
        val notes = ArrayList<String>()
        val located = LinkedHashMap<String, LocatedModel>()
        for (v in spec.variants) {
            val l = try {
                ModelLocator.locate(ctx, v)
            } catch (t: Throwable) {
                notes += "${v.id}: ${t.brief()}"
                null
            }
            when {
                l == null -> notes += "${v.id}: ${v.fileName} not found (external, imported or bundled)"
                !l.sha256Ok -> notes += "${v.id}: ${l.note}"
                else -> {
                    located[v.id] = l
                    l.note?.let { notes += "${v.id}: $it" }
                }
            }
        }
        val usable = spec.variants.filter { it.id in located }
        check(usable.any { !it.int8 }) { "${spec.name}: no usable float variant - ${notes.joinToString("; ")}" }
        val set = CandidateSet(usable)
        val key = CacheKeys.decisionKey(
            spec.name,
            usable.map { it.id to (located.getValue(it.id).actualSha256 ?: it.sha256) },
            AppInfo.versionCode(ctx),
            AppInfo.fingerprint(),
            AppInfo.nativeStamp(ctx),
        )

        var report: AcceleratorReport? = if (rebenchmark) null else store.decision(key)?.copy(fromCache = true)
        if (report != null) {
            val why = when {
                set.byLabel(report.chosen) == null -> "unknown candidate '${report.chosen}'"
                store.isCrashed(key, report.chosen) -> "${report.chosen} crashed since"
                else -> null
            }
            if (why != null) {
                RtLog.w("${spec.name}: cached decision dropped ($why); benchmarking again")
                report = null
            } else {
                RtLog.i("${spec.name}: cached decision: ${AccelRules.describe(report)}")
            }
        }
        val decided = report ?: benchmark(ctx, spec, set, located, key, store, notes)

        val chosen = set.byLabel(decided.chosen) ?: set.referenceCpu
        var target = chosen
        var forcedNote: String? = null
        if (forceAccel != null) {
            val (t, n) = forcedTarget(set, decided, chosen, forceAccel, store, key)
            target = t
            forcedNote = n
        }
        return instantiate(ctx, spec, set, located, key, store, decided, target, forceAccel != null, forcedNote)
    }

    private fun validate(spec: BackboneSpec) {
        require(spec.inputSize > 0 && spec.gh > 0 && spec.gw > 0 && spec.dim > 0) { "${spec.name}: bad sizes $spec" }
        require(spec.variants.isNotEmpty()) { "${spec.name}: no variants" }
        require(spec.variants.map { it.id }.toSet().size == spec.variants.size) { "${spec.name}: duplicate variant ids" }
        require(spec.variants.map { it.fileName }.toSet().size == spec.variants.size) { "${spec.name}: duplicate variant files" }
    }

    // --------------------------------------------------------------------------------------------------- benchmark
    private suspend fun benchmark(
        ctx: Context,
        spec: BackboneSpec,
        set: CandidateSet,
        located: Map<String, LocatedModel>,
        key: String,
        store: DecisionStore,
        locNotes: List<String>,
    ): AcceleratorReport {
        val dir = File(ctx.filesDir, "accel/probe").apply { mkdirs() }
        val soc = NpuEnv.socModel
        val npuOk = NpuEnv.npuSocSupported()
        val ms = ArrayList<Measurement>()
        val runNotes = ArrayList<String>()
        var ref: FloatArray? = null
        var probeProblem: String? = null
        val startedAt = System.currentTimeMillis()
        RtLog.i("${spec.name}: benchmarking ${set.runOrder.joinToString { it.label }} on ${NpuEnv.describe()}")
        ProbeClient(ctx).use { probe ->
            for (cand in set.runOrder) {
                val loc = located.getValue(cand.variant.id)
                var step: Step = when {
                    cand.accel == Accel.NPU && !npuOk ->
                        Step(Measurement(cand, error = "not attempted: SoC '${soc.ifEmpty { "?" }}' is not SM8850*"), null, null)
                    cand.accel != Accel.CPU && store.isCrashed(key, cand.label) ->
                        Step(Measurement(cand, error = "crashed during an earlier attempt - skipped (Runtime.forgetCrashes to retry)"), null, null)
                    !cand.isReference && ref == null ->
                        Step(Measurement(cand, error = "not attempted: no reference output"), null, null)
                    probeProblem != null || store.isCrashed(key, cand.label) ->
                        if (cand.accel == Accel.CPU) {
                            inProcess(ctx, spec, cand, loc, ref)
                        } else {
                            Step(Measurement(cand, error = "not attempted: probe unavailable ($probeProblem)"), null, null)
                        }
                    else -> viaProbe(ctx, probe, spec, cand, loc, ref, key, store, dir)
                }
                if (step.probeProblem != null) {
                    probeProblem = step.probeProblem
                    if (cand.accel == Accel.CPU) step = inProcess(ctx, spec, cand, loc, ref)
                }
                if (cand.isReference && step.output == null) {
                    // The reference must exist: measure it here (CPU in the app process is safe).
                    runNotes += "reference ${cand.label} failed in the probe (${step.measurement.error}); measured in the app process"
                    step = inProcess(ctx, spec, cand, loc, null, rethrow = true)
                }
                if (cand.isReference) {
                    val out = step.output
                    check(out != null && out.all { it.isFinite() }) {
                        "${spec.name}: the CPU reference output of ${cand.variant.id} is not finite - the model file is broken"
                    }
                    ref = out
                }
                ms += step.measurement
            }
        }
        dir.listFiles()?.forEach { it.delete() }
        val d = AccelRules.decide(set, ms)
        val device = "SoC ${NpuEnv.socManufacturer.ifEmpty { "?" }}/${soc.ifEmpty { "?" }}"
        val note = (listOf(device) + locNotes + runNotes + listOfNotNull(probeProblem?.let { "probe unavailable ($it): accelerators not tried" }))
            .joinToString("; ")
        val report = AcceleratorReport(
            model = spec.name,
            chosen = d.chosen.label,
            accel = d.chosen.accel,
            modelId = d.chosen.variant.id,
            badge = d.badge,
            speedupVsCpu = d.speedupVsCpu,
            results = d.results,
            fromCache = false,
            measuredAtMs = startedAt,
            cacheKey = key,
            note = note,
        )
        // A decision made without the probe is not cached: the next start tries the accelerators again.
        if (probeProblem == null) store.saveDecision(key, report)
        RtLog.i("${spec.name}: ${AccelRules.describe(report)}")
        return report
    }

    private fun inProcess(
        ctx: Context,
        spec: BackboneSpec,
        cand: Candidate,
        loc: LocatedModel,
        ref: FloatArray?,
        rethrow: Boolean = false,
    ): Step = try {
        val r = BackboneBench.measure(
            ctx, loc.source, requireNotNull(loc.path), Accel.CPU, spec.inputSize, spec.gh * spec.gw * spec.dim, spec.dim,
            AccelRules.WARMUP_RUNS, AccelRules.TIMED_RUNS,
        )
        val cmp = if (ref == null) null else AccelRules.compare(r.patches, ref, spec.dim)
        Step(Measurement(cand, r.timesMs, r.loadMs, cmp), r.patches, null)
    } catch (t: Throwable) {
        if (rethrow) throw IllegalStateException("${spec.name}: cannot run ${loc.asset.fileName} even on the CPU: ${t.brief()}", t)
        Step(Measurement(cand, error = "failed in the app process: ${t.brief()}"), null, null)
    }

    private suspend fun viaProbe(
        ctx: Context,
        probe: ProbeClient,
        spec: BackboneSpec,
        cand: Candidate,
        loc: LocatedModel,
        ref: FloatArray?,
        key: String,
        store: DecisionStore,
        dir: File,
    ): Step {
        val out = File(dir, cand.label.replace(Regex("[^A-Za-z0-9]+"), "_") + ".f32")
        out.delete()
        val timeout = probeTimeoutMs(loc.bytes)
        val req = ProbeRequest(
            probe.nextRequestId(), loc.source, requireNotNull(loc.path), cand.accel, spec.inputSize,
            spec.gh * spec.gw * spec.dim, spec.dim, AccelRules.WARMUP_RUNS, AccelRules.TIMED_RUNS, out.absolutePath,
        )
        store.writeTrying(key, cand.label, CrashGuard.Phase.PROBE)
        val t0 = System.currentTimeMillis()
        val outcome = try {
            probe.run(req, timeout)
        } catch (t: Throwable) {
            store.clearTrying()                      // a cancelled load must not be judged as a crash at the next start
            throw t
        }
        return when (val o = outcome) {
            is ProbeClient.Outcome.Ok -> {
                store.clearTrying()
                if (o.reply.ok) {
                    val output = try { FloatFiles.read(out) } catch (t: Throwable) { null }
                    out.delete()
                    if (output == null) {
                        Step(Measurement(cand, error = "probe output unreadable"), null, null)
                    } else {
                        val cmp = if (ref == null) null else AccelRules.compare(output, ref, spec.dim)
                        Step(Measurement(cand, o.reply.timesMs, o.reply.loadMs, cmp), output, null)
                    }
                } else {
                    Step(Measurement(cand, error = o.reply.error ?: "failed in the probe"), null, null)
                }
            }
            is ProbeClient.Outcome.Died -> {
                val reason = awaitExitReason(ctx, o.pid, t0)
                val before = store.strikes(key, cand.label)
                val after = CrashGuard.strikesAfterProbeDeath(before, CrashGuard.classify(reason), timedOut = false)
                store.setStrikes(key, cand.label, after)
                store.clearTrying()
                val tail = if (CrashGuard.isCrashed(after)) "skipped from now on" else "will be retried once"
                RtLog.e("${spec.name}: probe died (${CrashGuard.reasonName(reason)}, ${o.how}) running ${cand.label}; $tail")
                Step(Measurement(cand, error = "probe process died (${CrashGuard.reasonName(reason)}) while running ${cand.label} - $tail"), null, null)
            }
            ProbeClient.Outcome.TimedOut -> {
                store.setStrikes(key, cand.label, CrashGuard.CRASHED)
                store.clearTrying()
                Step(Measurement(cand, error = "no answer within ${timeout / 1000} s (hung?) - probe killed; skipped from now on"), null, null)
            }
            is ProbeClient.Outcome.Unavailable -> {
                store.clearTrying()
                Step(Measurement(cand, error = "not attempted: probe unavailable (${o.reason})"), null, o.reason)
            }
        }
    }

    private suspend fun awaitExitReason(ctx: Context, pid: Int, since: Long): Int? {
        if (pid <= 0) return null
        repeat(8) {
            DecisionStore.exitReason(ctx, pid, since)?.let { return it }
            delay(250)
        }
        return null
    }

    // ------------------------------------------------------------------------------------------------------ forced
    /** A/B toggle: same variant as the decision, on [accel]; refused (-> CPU, same variant) unless benchmarked OK. */
    private fun forcedTarget(
        set: CandidateSet,
        report: AcceleratorReport,
        chosen: Candidate,
        accel: Accel,
        store: DecisionStore,
        key: String,
    ): Pair<Candidate, String?> {
        val want = set.candidate(accel, chosen.variant)
        if (accel == Accel.CPU) return want to null
        val r = report.results.firstOrNull { it.name == want.label }
        val refuse = when {
            store.isCrashed(key, want.label) -> "crashed during an earlier attempt"
            r == null -> "not benchmarked on this device"
            r.error != null -> r.error
            else -> null
        } ?: return want to null
        val cpu = set.cpu(chosen.variant)
        return cpu to "forced ${want.label} refused ($refuse) - running ${cpu.label}"
    }

    private fun reportFor(set: CandidateSet, report: AcceleratorReport, target: Candidate, forced: Boolean, forcedNote: String?): AcceleratorReport {
        if (!forced) return report
        val r = report.results.firstOrNull { it.name == target.label }
        val base = report.results.firstOrNull { it.name == target.baselineLabel }
        val speedup = if (target.accel != Accel.CPU && r?.medianMs != null && base?.medianMs != null) base.medianMs / r.medianMs else null
        return report.copy(
            chosen = target.label,
            accel = target.accel,
            modelId = target.variant.id,
            badge = AccelRules.forcedBadge(target, r, base),
            speedupVsCpu = speedup,
            forced = true,
            note = joinNotes(forcedNote, "A/B: ${target.label} forced (benchmark chose ${report.chosen})"),
        )
    }

    // ------------------------------------------------------------------------------------------------- instantiate
    private fun instantiate(
        ctx: Context,
        spec: BackboneSpec,
        set: CandidateSet,
        located: Map<String, LocatedModel>,
        key: String,
        store: DecisionStore,
        report: AcceleratorReport,
        target: Candidate,
        forced: Boolean,
        forcedNote: String?,
    ): FeatureBackbone {
        val loc = located.getValue(target.variant.id)
        val guard = target.accel != Accel.CPU
        val needTiming = report.results.firstOrNull { it.name == target.label }?.medianMs == null
        var model: CompiledModel? = null
        var impl: BackboneImpl? = null
        if (guard) store.writeTrying(key, target.label, CrashGuard.Phase.MAIN)
        val failure: Throwable? = try {
            val t0 = nowMs()
            model = LiteRtModels.create(ctx, loc, target.accel)
            val createMs = nowMs() - t0
            val rep = reportFor(set, report, target, forced, forcedNote)
            impl = BackboneImpl(ctx, spec, loc, target.accel, model, rep) { cur, cause ->
                runtimeFallback(set, key, store, forced, cur, cause, decisionNote = report.note)
            }
            model = null // owned by impl now
            val times = impl.warmUp(if (needTiming) 5 else 1)
            val t = AccelRules.timing(times)
            // In-app create time (NPU: JIT result from the compiler cache if the probe compiled it) - compare with the probe's loadMs.
            val loaded = "app load ${AccelRules.fmtMs(createMs)} ms, warm run ${t?.medianMs?.let { AccelRules.fmtMs(it) } ?: "?"} ms"
            impl.setReport(
                if (needTiming && t != null) {
                    impl.report.copy(
                        badge = AccelRules.badge(target, t.medianMs, null) + if (forced) " (forced)" else "",
                        note = joinNotes(impl.report.note, "timing measured in the app (${times.size} runs)", loaded),
                    )
                } else {
                    impl.report.copy(note = joinNotes(impl.report.note, loaded))
                },
            )
            null
        } catch (t: Throwable) {
            t
        }
        if (guard) store.clearTrying()
        if (failure == null) {
            val ok = impl!!
            RtLog.i("${spec.name}: ready on ${target.label} (${loc.source}): ${ok.report.badge}")
            return ok
        }
        try { impl?.close() } catch (_: Throwable) { }
        LiteRtModels.closeQuietly(model)
        if (!guard) {
            throw IllegalStateException("${spec.name}: cannot load ${target.label} from ${loc.source} ${loc.path}: ${failure.brief()}", failure)
        }
        RtLog.e("${spec.name}: ${target.label} failed in the app process: ${failure.brief()}")
        val cpu = set.cpu(target.variant)
        val results = report.results.map {
            if (it.name == target.label) it.copy(ok = false, error = "failed to load in the app process: ${failure.brief()}") else it
        }
        val cpuRes = results.firstOrNull { it.name == cpu.label }
        val why = "${target.label} failed to load in the app (${failure.brief()}); using ${cpu.label}"
        val updated = report.copy(
            chosen = cpu.label, accel = Accel.CPU, modelId = cpu.variant.id,
            badge = AccelRules.badge(cpu, cpuRes?.medianMs, null) + " (fallback)",
            speedupVsCpu = null, results = results, note = joinNotes(report.note, why),
        )
        if (!forced) store.saveDecision(key, updated)
        return instantiate(ctx, spec, set, located, key, store, updated, cpu, forced, if (forced) joinNotes(forcedNote, why) else null)
    }

    /** The chosen accelerator failed at run time: that instance is on the CPU (same variant) for good; cache updated. */
    private fun runtimeFallback(
        set: CandidateSet,
        key: String,
        store: DecisionStore,
        forced: Boolean,
        rep: AcceleratorReport,
        cause: Throwable,
        decisionNote: String?,
    ): AcceleratorReport {
        val failed = set.byLabel(rep.chosen)
        val variant = failed?.variant ?: set.reference
        val cpu = set.cpu(variant)
        val results = rep.results.map { if (it.name == rep.chosen) it.copy(ok = false, error = "failed at run time: ${cause.brief()}") else it }
        val cpuRes = results.firstOrNull { it.name == cpu.label }
        val why = "${rep.chosen} failed at run time (${cause.brief()}); switched to ${cpu.label} for good"
        val updated = rep.copy(
            chosen = cpu.label, accel = Accel.CPU, modelId = variant.id,
            badge = AccelRules.badge(cpu, cpuRes?.medianMs, null) + " (fallback)",
            speedupVsCpu = null, results = results,
            note = joinNotes(rep.note, why),
        )
        if (!forced) {
            try { store.saveDecision(key, updated.copy(note = joinNotes(decisionNote, why))) } catch (_: Throwable) { }
        }
        return updated
    }
}

internal fun joinNotes(vararg parts: String?): String? = parts.filterNotNull().filter { it.isNotBlank() }.joinToString("; ").ifEmpty { null }
