package com.kaizeneye.v2.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * REPLAY: feeds a recorded clip through the identical pipeline (plan B2/B4). Two clip formats:
 *  - a JPEG-sequence folder: `frames/NNNNNN.jpg` + `timestamps.json` (ms per frame) [+ optional `script.json` with
 *    `"rotationDegrees"`] — the laptop-generated self-test clip (testdata/replay_synth) uses this;
 *  - an `.mp4` recorded in-app by CameraX (sensor orientation; rotation from the container metadata).
 * Frames are decoded in timestamp order and handed over with their OWN timestamps, so a replay is deterministic and — with
 * [paced] = false — runs as fast as the pipeline allows (regression mode). [paced] = true sleeps to real time (demo mode).
 */
class ReplayFrameSource(private val clip: File, private val paced: Boolean) : FrameSource {

    override val kind = FrameSourceKind.REPLAY

    data class Progress(val frames: Int, val total: Int, val done: Boolean, val error: String? = null)

    @Volatile private var consumer: FrameConsumer? = null
    private val running = AtomicBoolean(false)
    private var thread: Thread? = null
    private val _progress = MutableStateFlow(Progress(0, 0, false))
    val progress: StateFlow<Progress> = _progress

    override fun setConsumer(consumer: FrameConsumer?) {
        this.consumer = consumer
    }

    fun start(onFinished: (Progress) -> Unit = {}) {
        if (!running.compareAndSet(false, true)) return
        thread = Thread({
            val result = try {
                if (clip.isDirectory) playJpegSequence() else playMp4()
            } catch (t: Throwable) {
                Log.e(TAG, "replay failed", t)
                _progress.value.copy(done = true, error = t.message ?: t.javaClass.simpleName)
            }
            _progress.value = result
            running.set(false)
            onFinished(result)
        }, "kz-replay").apply { priority = Thread.MAX_PRIORITY; start() }
    }

    fun stop() {
        running.set(false)
        thread?.interrupt()
    }

    /** Blocks until the replay thread has finished (used by the self-test). */
    fun join(timeoutMs: Long) {
        thread?.join(timeoutMs)
    }

    private fun playJpegSequence(): Progress {
        val ts = JSONArray(File(clip, "timestamps.json").readText())
        val frames = File(clip, "frames").listFiles { f -> f.name.endsWith(".jpg") }?.sortedBy { it.name } ?: emptyList()
        val n = minOf(ts.length(), frames.size)
        val rotation = File(clip, "script.json").takeIf { it.isFile }
            ?.let { JSONObject(it.readText()).optInt("rotationDegrees", 0) } ?: 0
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        var data = ByteArray(0)
        val clock = Pacer(paced)
        for (i in 0 until n) {
            if (!running.get()) return Progress(i, n, true, "stopped")
            val t = ts.getLong(i)
            val bmp = BitmapFactory.decodeFile(frames[i].absolutePath, opts) ?: error("cannot decode ${frames[i].name}")
            val need = bmp.rowBytes * bmp.height
            if (data.size < need) data = ByteArray(need)
            bmp.copyPixelsToBuffer(ByteBuffer.wrap(data)) // ARGB_8888 = R,G,B,A byte order, same as CameraX RGBA_8888
            clock.waitFor(t)
            consumer?.onFrame(CameraFrame(t, bmp.width, bmp.height, bmp.rowBytes, data, rotation, kind, i.toLong()))
            bmp.recycle()
            _progress.value = Progress(i + 1, n, false)
        }
        return Progress(n, n, true)
    }

    private fun playMp4(): Progress {
        val times = videoTimestampsMs(clip)
        val mmr = MediaMetadataRetriever()
        try {
            mmr.setDataSource(clip.absolutePath)
            val count = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: times.size
            val durMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val rotation = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            val n = count
            val params = MediaMetadataRetriever.BitmapParams().apply { preferredConfig = Bitmap.Config.ARGB_8888 }
            var data = ByteArray(0)
            val clock = Pacer(paced)
            var i = 0
            while (i < n) {
                if (!running.get()) return Progress(i, n, true, "stopped")
                val batch = minOf(BATCH, n - i)
                val bitmaps = mmr.getFramesAtIndex(i, batch, params)
                for ((k, bmp) in bitmaps.withIndex()) {
                    val idx = i + k
                    val t = if (times.size == n) times[idx] else idx * durMs / maxOf(1, n)
                    val need = bmp.rowBytes * bmp.height
                    if (data.size < need) data = ByteArray(need)
                    bmp.copyPixelsToBuffer(ByteBuffer.wrap(data))
                    clock.waitFor(t)
                    consumer?.onFrame(CameraFrame(t, bmp.width, bmp.height, bmp.rowBytes, data, rotation, kind, idx.toLong()))
                    bmp.recycle()
                }
                i += batch
                _progress.value = Progress(i, n, false)
            }
            return Progress(n, n, true)
        } finally {
            mmr.release()
        }
    }

    private class Pacer(private val paced: Boolean) {
        private var t0Clip = Long.MIN_VALUE
        private var t0Wall = 0L
        fun waitFor(tClip: Long) {
            if (!paced) return
            if (t0Clip == Long.MIN_VALUE) {
                t0Clip = tClip
                t0Wall = System.nanoTime() / 1_000_000L
                return
            }
            val due = t0Wall + (tClip - t0Clip)
            val now = System.nanoTime() / 1_000_000L
            if (due > now) Thread.sleep(due - now)
        }
    }

    companion object {
        private const val TAG = "KaizenReplay"
        private const val BATCH = 4

        /** Presentation timestamps (ms, ascending) of the first video track. */
        fun videoTimestampsMs(file: File): List<Long> {
            val ex = MediaExtractor()
            try {
                ex.setDataSource(file.absolutePath)
                val track = (0 until ex.trackCount).firstOrNull {
                    ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
                } ?: return emptyList()
                ex.selectTrack(track)
                val out = ArrayList<Long>()
                while (true) {
                    val t = ex.sampleTime
                    if (t < 0) break
                    out += t / 1000L
                    if (!ex.advance()) break
                }
                out.sort()
                return out
            } catch (t: Throwable) {
                return emptyList()
            } finally {
                ex.release()
            }
        }

        /** A clip is either an .mp4 file or a folder with timestamps.json + frames/. */
        fun isClip(f: File): Boolean =
            (f.isFile && f.name.endsWith(".mp4")) || (f.isDirectory && File(f, "timestamps.json").isFile && File(f, "frames").isDirectory)
    }
}
