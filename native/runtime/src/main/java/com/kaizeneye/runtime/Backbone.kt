package com.kaizeneye.runtime

import android.content.Context
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * A loaded backbone model in the app process. embed() is serialised (one LiteRT CompiledModel + one set of TensorBuffers
 * reused), converts the crop bytes to float32 0..255 NHWC, runs, reads the output whose element count is gh*gw*dim (other
 * outputs, e.g. DINOv2's CLS [1,dim], are ignored unless asked for) and optionally L2-normalises each patch vector.
 * If the accelerator fails at run time, the instance switches to the CPU for good with the SAME model variant (modelId and
 * everything built from its features stay valid) and the decision cache is updated through [onRuntimeFailure].
 */
internal class BackboneImpl(
    private val context: Context,
    override val spec: BackboneSpec,
    private val located: LocatedModel,
    accel: Accel,
    model: CompiledModel,
    initialReport: AcceleratorReport,
    private val onRuntimeFailure: (AcceleratorReport, Throwable) -> AcceleratorReport,
) : FeatureBackbone {
    private val lock = Any()
    private val n = spec.inputSize
    private val patchLen = spec.gh * spec.gw * spec.dim
    private val input = FloatArray(n * n * 3)
    private var model: CompiledModel = model
    private var ins: List<TensorBuffer>
    private var outs: List<TensorBuffer>
    private var patchIndex = -1
    private var clsIndex = -1
    private var closed = false

    @Volatile private var current: AcceleratorReport = initialReport

    @Volatile var accel: Accel = accel
        private set

    init {
        val i = model.createInputBuffers()
        val o = try {
            model.createOutputBuffers()
        } catch (t: Throwable) {
            LiteRtModels.closeQuietly(i)
            throw t
        }
        if (i.size != 1 || o.isEmpty()) {
            LiteRtModels.closeQuietly(i)
            LiteRtModels.closeQuietly(o)
            throw IllegalStateException("${located.asset.fileName}: backbone must have 1 input and >= 1 output (has ${i.size} / ${o.size})")
        }
        ins = i
        outs = o
    }

    override val modelId: String get() = located.asset.id
    override val sha256: String get() = located.actualSha256 ?: located.asset.sha256.lowercase()
    override val report: AcceleratorReport get() = current
    override val hasCls: Boolean get() = clsIndex >= 0

    /** The file this instance runs (source + path + digest). */
    val file: LocatedModel get() = located

    /**
     * First run on the benchmark image: identifies the patch/CLS outputs by element count (throws a clear error if no
     * output has gh*gw*dim values), then [timed] timed runs. Returns their times (ms).
     */
    fun warmUp(timed: Int): DoubleArray = synchronized(lock) {
        val img = BenchInputs.image(n)
        System.arraycopy(img, 0, input, 0, input.size)
        ins[0].writeFloat(input)
        model.run(ins, outs)
        identifyOutputsLocked()
        DoubleArray(timed.coerceAtLeast(0)) {
            val t = nowMs()
            runLocked(false)
            nowMs() - t
        }
    }

    private fun identifyOutputsLocked() {
        val sizes = outs.map { b -> try { b.readFloat().size } catch (_: Throwable) { -1 } }
        val (pi, ci) = OutputPick.pick(sizes, patchLen, spec.dim)
        patchIndex = pi
        clsIndex = ci
        if (sizes.size > 1) RtLog.i("${spec.name}/${located.asset.id}: output sizes $sizes -> patches #$pi, cls ${if (ci >= 0) "#$ci" else "none"}")
    }

    override fun embed(rgb: ByteArray): FloatArray = embedBytes(rgb, false).first

    override fun embedWithCls(rgb: ByteArray): Pair<FloatArray, FloatArray?> = embedBytes(rgb, true)

    private fun embedBytes(rgb: ByteArray, wantCls: Boolean): Pair<FloatArray, FloatArray?> = synchronized(lock) {
        check(!closed) { "backbone ${spec.name} is closed" }
        require(rgb.size == input.size) { "rgb has ${rgb.size} bytes, expected ${input.size} (${n}x${n}x3)" }
        for (i in rgb.indices) input[i] = (rgb[i].toInt() and 0xFF).toFloat()
        val out = runWithFallbackLocked(wantCls)
        if (spec.l2NormalizePatches) FeatureMath.l2NormalizeRows(out.first, spec.dim)
        out
    }

    /** Raw float input (self-tests): no rounding, no normalisation. */
    fun runFloat(x: FloatArray): FloatArray = synchronized(lock) {
        check(!closed) { "backbone ${spec.name} is closed" }
        require(x.size == input.size) { "input has ${x.size} values, expected ${input.size}" }
        System.arraycopy(x, 0, input, 0, input.size)
        runWithFallbackLocked(false).first
    }

    private fun runLocked(wantCls: Boolean): Pair<FloatArray, FloatArray?> {
        if (patchIndex < 0) {
            ins[0].writeFloat(input)
            model.run(ins, outs)
            identifyOutputsLocked()
        }
        ins[0].writeFloat(input)
        model.run(ins, outs)
        val p = outs[patchIndex].readFloat()
        check(p.size == patchLen) { "backbone output has ${p.size} values, expected $patchLen" }
        val c = if (wantCls && clsIndex >= 0) outs[clsIndex].readFloat() else null
        return p to c
    }

    private fun runWithFallbackLocked(wantCls: Boolean): Pair<FloatArray, FloatArray?> {
        try {
            return runLocked(wantCls)
        } catch (t: Throwable) {
            if (accel == Accel.CPU) throw t
            // One transient accelerator error (a driver hiccup) is retried before the instance falls back to CPU for good.
            try {
                patchIndex = -1
                return runLocked(wantCls)
            } catch (t2: Throwable) {
                switchToCpuLocked(t2)
                return runLocked(wantCls)
            }
        }
    }

    /** The accelerator failed at run time: CPU, same variant, for good. */
    private fun switchToCpuLocked(cause: Throwable) {
        RtLog.e("${current.chosen} failed at run time for ${spec.name} (${cause.brief()}); switching this instance to CPU for good", cause)
        val cpu = LiteRtModels.create(context, located, Accel.CPU)
        val newIns: List<TensorBuffer>
        val newOuts: List<TensorBuffer>
        try {
            newIns = cpu.createInputBuffers()
            newOuts = cpu.createOutputBuffers()
        } catch (t: Throwable) {
            LiteRtModels.closeQuietly(cpu)
            throw t
        }
        LiteRtModels.closeQuietly(ins)
        LiteRtModels.closeQuietly(outs)
        LiteRtModels.closeQuietly(model)
        model = cpu
        ins = newIns
        outs = newOuts
        patchIndex = -1
        clsIndex = -1
        accel = Accel.CPU
        current = try {
            onRuntimeFailure(current, cause)
        } catch (t: Throwable) {
            current.copy(accel = Accel.CPU, note = "run-time fallback to CPU (${cause.brief()})")
        }
    }

    /** For the loader: replace the report (e.g. in-app timing for a forced candidate). */
    fun setReport(r: AcceleratorReport) {
        current = r
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            LiteRtModels.closeQuietly(ins)
            LiteRtModels.closeQuietly(outs)
            LiteRtModels.closeQuietly(model)
        }
    }
}
