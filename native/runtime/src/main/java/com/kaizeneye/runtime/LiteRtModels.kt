package com.kaizeneye.runtime

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.NpuCompatibilityChecker
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * LiteRT environments of THIS process. Two, on purpose:
 *  - plain: CPU/GPU models (k-NN graphs, CPU backbones) - never loads the Qualcomm dispatch library;
 *  - npu:   created only when an NPU model is created (in ":probe" first; in the app only after the probe survived it, under
 *           the crash-guard marker). BuiltinNpuAcceleratorProvider with our SoC checker (SOC_MODEL startsWith "SM8850";
 *           LiteRT's default demands exactly "SM8850"), dispatch + compiler-plugin dirs = nativeLibraryDir, JIT compiler
 *           cache on (context.cacheDir) so the app process reuses what the probe compiled.
 * Environments live for the whole process (models keep a reference); they are never closed.
 */
internal object LiteRtEnv {
    @Volatile private var plain: Environment? = null
    @Volatile private var npu: Environment? = null

    fun plain(): Environment {
        plain?.let { return it }
        synchronized(this) {
            plain?.let { return it }
            return Environment.create().also { plain = it }
        }
    }

    fun npu(context: Context): Environment {
        npu?.let { return it }
        synchronized(this) {
            npu?.let { return it }
            NpuEnv.ensure(context)
            val dir = context.applicationInfo.nativeLibraryDir
            val checker = object : NpuCompatibilityChecker {
                override fun isDeviceSupported(): Boolean = NpuSoc.supported(NpuEnv.socModel)
            }
            val env = Environment.create(
                context = context,
                npuAcceleratorProvider = BuiltinNpuAcceleratorProvider(context, checker),
                options = mapOf(
                    Environment.Option.DispatchLibraryDir to dir,
                    Environment.Option.CompilerPluginLibraryDir to dir,
                ),
                enableCompilerCache = true,
            )
            RtLog.i("LiteRT NPU environment created (dispatch dir $dir); available: ${available(env)}")
            npu = env
            return env
        }
    }

    fun forAccel(context: Context, accel: Accel): Environment = if (accel == Accel.NPU) npu(context) else plain()

    fun available(env: Environment): String = try {
        env.getAvailableAccelerators().joinToString(",") { it.name }
    } catch (t: Throwable) {
        "unknown (${t.brief()})"
    }
}

internal object LiteRtModels {
    const val CPU_THREADS = 4

    /** Options per accelerator: CPU 4 threads; GPU full FP32 precision; NPU HTP SUSTAINED_HIGH_PERFORMANCE (fixed at load). */
    fun options(accel: Accel): CompiledModel.Options = when (accel) {
        Accel.CPU -> CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(numThreads = CPU_THREADS)
        }
        Accel.GPU -> CompiledModel.Options(Accelerator.GPU).apply {
            gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
        }
        Accel.NPU -> CompiledModel.Options(Accelerator.NPU).apply {
            qualcommOptions = CompiledModel.QualcommOptions(
                htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.SUSTAINED_HIGH_PERFORMANCE,
            )
        }
    }

    /** [source] "bundled" -> [path] is the asset name ("models/<file>"); otherwise an absolute file path. */
    fun create(context: Context, source: String, path: String, accel: Accel): CompiledModel {
        val env = LiteRtEnv.forAccel(context, accel)
        val opts = options(accel)
        return if (source == ModelLocator.SOURCE_BUNDLED) {
            CompiledModel.create(context.assets, path, opts, env)
        } else {
            CompiledModel.create(path, opts, env)
        }
    }

    fun create(context: Context, located: LocatedModel, accel: Accel): CompiledModel =
        create(context, located.source, requireNotNull(located.path) { "model ${located.asset.fileName} has no path" }, accel)

    /** Whole model file bytes (small models only: the k-NN graphs are ~2 KB). */
    fun readBytes(context: Context, located: LocatedModel, maxBytes: Long = 32L shl 20): ByteArray {
        val path = requireNotNull(located.path) { "model ${located.asset.fileName} has no path" }
        require(located.bytes <= maxBytes) { "${located.asset.fileName} is ${located.bytes} bytes; readBytes is for small models" }
        return if (located.source == ModelLocator.SOURCE_BUNDLED) {
            context.assets.open(path).use { it.readBytes() }
        } else {
            File(path).readBytes()
        }
    }

    fun closeQuietly(buffers: List<TensorBuffer>?) {
        buffers?.forEach { b -> try { b.close() } catch (_: Throwable) { } }
    }

    fun closeQuietly(model: CompiledModel?) {
        try { model?.close() } catch (_: Throwable) { }
    }
}

/**
 * The backbone benchmark protocol, shared by the ":probe" process and the in-app fallback: create the model on [accel], run
 * the fixed benchmark image (BenchInputs.image) once to find the outputs by element count (patch output = gh*gw*dim values,
 * optional CLS = dim values), then [warmup]-1 more warm-ups and [runs] timed runs, each timed run = write input + run + read
 * the patch output (what FeatureBackbone.embed costs).
 */
internal object BackboneBench {
    class Result(
        val loadMs: Double,
        val timesMs: DoubleArray,
        val patches: FloatArray,
        val patchIndex: Int,
        val clsIndex: Int,
        val outputs: Int,
        val available: String,
    )

    fun measure(
        context: Context,
        source: String,
        path: String,
        accel: Accel,
        inputSize: Int,
        patchLen: Int,
        clsLen: Int,
        warmup: Int,
        runs: Int,
    ): Result {
        val env = LiteRtEnv.forAccel(context, accel)
        val t0 = nowMs()
        val model = LiteRtModels.create(context, source, path, accel)
        val loadMs = nowMs() - t0
        var ins: List<TensorBuffer>? = null
        var outs: List<TensorBuffer>? = null
        try {
            ins = model.createInputBuffers()
            outs = model.createOutputBuffers()
            check(ins.size == 1) { "backbone must have 1 input, this model has ${ins.size}" }
            check(outs.isNotEmpty()) { "backbone has no outputs" }
            val input = BenchInputs.image(inputSize)
            ins[0].writeFloat(input)
            model.run(ins, outs)
            val sizes = outs.map { b -> try { b.readFloat().size } catch (_: Throwable) { -1 } }
            val (pi, ci) = OutputPick.pick(sizes, patchLen, clsLen)
            repeat((warmup - 1).coerceAtLeast(0)) {
                ins[0].writeFloat(input)
                model.run(ins, outs)
                outs[pi].readFloat()
            }
            val times = DoubleArray(runs.coerceAtLeast(1))
            var last = FloatArray(0)
            for (i in times.indices) {
                val t = nowMs()
                ins[0].writeFloat(input)
                model.run(ins, outs)
                last = outs[pi].readFloat()
                times[i] = nowMs() - t
            }
            return Result(loadMs, times, last, pi, ci, outs.size, LiteRtEnv.available(env))
        } finally {
            LiteRtModels.closeQuietly(ins)
            LiteRtModels.closeQuietly(outs)
            LiteRtModels.closeQuietly(model)
        }
    }
}

/** Pure: which model output is the patch map and which the CLS vector, by element count (names are not reliable). */
internal object OutputPick {
    /** Returns (patchIndex, clsIndex or -1). Throws a clear error when no output has [patchLen] values. */
    fun pick(sizes: List<Int>, patchLen: Int, clsLen: Int): Pair<Int, Int> {
        val pi = sizes.indexOf(patchLen)
        check(pi >= 0) { "no model output has gh*gw*dim = $patchLen values (output sizes: $sizes)" }
        val ci = if (clsLen > 0) sizes.indices.firstOrNull { it != pi && sizes[it] == clsLen } ?: -1 else -1
        return pi to ci
    }
}
