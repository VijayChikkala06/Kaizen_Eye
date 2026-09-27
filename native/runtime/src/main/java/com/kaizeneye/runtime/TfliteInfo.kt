package com.kaizeneye.runtime

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal read-only view of a .tflite FlatBuffer (schema.fbs): the main subgraph's input/output tensors (name, shape, type)
 * and the SignatureDefs. LiteRT 2.2.0's Kotlin API can create buffers by input NAME but cannot list names or shapes, so the
 * k-NN runner reads them here and identifies feats/bankT/bankNorm by shape. Pure Kotlin (JVM-tested on the real models).
 *
 * Verified layouts (ai_edge_litert): knn_p1600_d128_k*.tflite has NO signature, inputs "feats" [1600,128], "bankT" [128,K],
 * "bankNorm" [K], output "Identity" [1600]; backbone_r18_320*.tflite has signature "serving_default" (input "image").
 */
internal class TfliteInfo(
    val inputs: List<Tensor>,
    val outputs: List<Tensor>,
    val signatures: List<Signature>,
) {
    /** [type] is the TFLite TensorType enum (0 = FLOAT32, 9 = INT8, ...). */
    class Tensor(val index: Int, val name: String, val shape: IntArray, val type: Int) {
        val elements: Long get() = shape.fold(1L) { a, b -> a * b }
        override fun toString(): String = "$name${shape.contentToString()}:${typeName(type)}"
    }

    class Signature(val key: String, val subgraph: Int, val inputs: List<Pair<String, Int>>, val outputs: List<Pair<String, Int>>) {
        override fun toString(): String = "$key(in=${inputs.map { it.first }}, out=${outputs.map { it.first }})"
    }

    /** First input whose shape equals [shape] exactly. */
    fun inputWithShape(vararg shape: Int): Tensor? = inputs.firstOrNull { it.shape.contentEquals(shape) }

    fun outputWithShape(vararg shape: Int): Tensor? = outputs.firstOrNull { it.shape.contentEquals(shape) }

    /** Signature-level name of a subgraph tensor (the name LiteRT uses for signature models), else the tensor name. */
    fun signatureInputName(t: Tensor): String =
        signatures.firstOrNull { it.subgraph == 0 }?.inputs?.firstOrNull { it.second == t.index }?.first ?: t.name

    fun signatureOutputName(t: Tensor): String =
        signatures.firstOrNull { it.subgraph == 0 }?.outputs?.firstOrNull { it.second == t.index }?.first ?: t.name

    override fun toString(): String = "inputs=$inputs outputs=$outputs signatures=$signatures"

    companion object {
        const val FLOAT32 = 0
        const val INT8 = 9

        fun typeName(t: Int): String = when (t) {
            0 -> "float32"; 1 -> "float16"; 2 -> "int32"; 3 -> "uint8"; 4 -> "int64"; 5 -> "string"; 6 -> "bool"
            7 -> "int16"; 9 -> "int8"; 10 -> "float64"; else -> "type$t"
        }

        fun parse(bytes: ByteArray): TfliteInfo = parse(ByteBuffer.wrap(bytes))

        /** Parse [buffer] (absolute positions from 0; not modified). Throws IllegalArgumentException on a malformed file. */
        fun parse(buffer: ByteBuffer): TfliteInfo {
            val fb = Fb(buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN))
            try {
                val model = fb.root()
                val subgraphs = fb.vector(model, 2) ?: throw IllegalArgumentException("no subgraphs")
                require(fb.len(subgraphs) > 0) { "no subgraphs" }
                val sg = fb.tableAt(subgraphs, 0)
                val tensorsVec = fb.vector(sg, 0) ?: throw IllegalArgumentException("no tensors")
                fun tensor(i: Int): Tensor {
                    require(i >= 0 && i < fb.len(tensorsVec)) { "tensor index $i out of range" }
                    val t = fb.tableAt(tensorsVec, i)
                    val shapeVec = fb.vector(t, 0)
                    val shape = if (shapeVec == null) IntArray(0) else IntArray(fb.len(shapeVec)) { fb.intAt(shapeVec, it) }
                    val type = fb.scalarByte(t, 1, 0)
                    val name = fb.string(t, 3) ?: ""
                    return Tensor(i, name, shape, type)
                }
                val inVec = fb.vector(sg, 1)
                val outVec = fb.vector(sg, 2)
                val inputs = if (inVec == null) emptyList() else List(fb.len(inVec)) { tensor(fb.intAt(inVec, it)) }
                val outputs = if (outVec == null) emptyList() else List(fb.len(outVec)) { tensor(fb.intAt(outVec, it)) }
                val sigs = ArrayList<Signature>()
                val sigVec = fb.vector(model, 7)
                if (sigVec != null) {
                    for (s in 0 until fb.len(sigVec)) {
                        val st = fb.tableAt(sigVec, s)
                        fun maps(field: Int): List<Pair<String, Int>> {
                            val v = fb.vector(st, field) ?: return emptyList()
                            return List(fb.len(v)) {
                                val m = fb.tableAt(v, it)
                                (fb.string(m, 0) ?: "") to fb.scalarInt(m, 1, 0)
                            }
                        }
                        sigs += Signature(fb.string(st, 2) ?: "", fb.scalarInt(st, 4, 0), maps(0), maps(1))
                    }
                }
                return TfliteInfo(inputs, outputs, sigs)
            } catch (e: IllegalArgumentException) {
                throw e
            } catch (e: RuntimeException) {       // IndexOutOfBounds etc. on a truncated/garbage file
                throw IllegalArgumentException("malformed .tflite flatbuffer: ${e.brief()}", e)
            }
        }
    }

    /** FlatBuffers primitives over absolute little-endian positions. */
    private class Fb(val bb: ByteBuffer) {
        private val limit = bb.limit()

        private fun check(pos: Int, size: Int) {
            if (pos < 0 || size < 0 || pos.toLong() + size > limit) throw IllegalArgumentException("offset $pos(+$size) outside the file ($limit bytes)")
        }

        fun i32(pos: Int): Int { check(pos, 4); return bb.getInt(pos) }

        fun u16(pos: Int): Int { check(pos, 2); return bb.getShort(pos).toInt() and 0xffff }

        fun u8(pos: Int): Int { check(pos, 1); return bb.get(pos).toInt() and 0xff }

        fun root(): Int = i32(0)

        /** Absolute position of [field]'s value in table [t], or 0 when the field is absent. */
        fun fieldPos(t: Int, field: Int): Int {
            val vt = t - i32(t)
            val vtSize = u16(vt)
            val o = 4 + 2 * field
            if (o + 2 > vtSize) return 0
            val off = u16(vt + o)
            return if (off == 0) 0 else t + off
        }

        private fun indirect(pos: Int): Int = pos + i32(pos)

        /** Position of a vector's length word (elements follow), or null when absent. */
        fun vector(t: Int, field: Int): Int? {
            val p = fieldPos(t, field)
            return if (p == 0) null else indirect(p)
        }

        fun len(vec: Int): Int {
            val n = i32(vec)
            if (n < 0 || n > (limit - vec) / 1) throw IllegalArgumentException("bad vector length $n")
            return n
        }

        fun tableAt(vec: Int, i: Int): Int = indirect(vec + 4 + 4 * i)

        fun intAt(vec: Int, i: Int): Int = i32(vec + 4 + 4 * i)

        fun string(t: Int, field: Int): String? {
            val s = vector(t, field) ?: return null
            val n = len(s)
            check(s + 4, n)
            val b = ByteArray(n)
            for (k in 0 until n) b[k] = bb.get(s + 4 + k)
            return String(b, Charsets.UTF_8)
        }

        fun scalarInt(t: Int, field: Int, default: Int): Int {
            val p = fieldPos(t, field)
            return if (p == 0) default else i32(p)
        }

        fun scalarByte(t: Int, field: Int, default: Int): Int {
            val p = fieldPos(t, field)
            return if (p == 0) default else u8(p)
        }
    }
}
