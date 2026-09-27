package com.kaizeneye.v2

import android.content.Context
import com.kaizeneye.v2.camera.CameraPreset
import org.json.JSONObject

/** Small persisted settings (SharedPreferences). Per-Twin values are keyed by Twin id. */
class Prefs(ctx: Context) {
    private val sp = ctx.getSharedPreferences("kaizen", Context.MODE_PRIVATE)

    var activeTwinId: String?
        get() = sp.getString("activeTwinId", null)
        set(v) = sp.edit().putString("activeTwinId", v).apply()

    /** Sensitivity multiplier on τ (plan: slider 0.8–2.0×, persisted per Twin). */
    fun sensitivity(twinId: String): Float = sp.getFloat("sens_$twinId", 1.0f)
    fun setSensitivity(twinId: String, v: Float) = sp.edit().putFloat("sens_$twinId", v.coerceIn(0.8f, 2.0f)).apply()

    /** Trigger: "LINE", "STEADY" or "BOTH". */
    var triggerMode: String
        get() = sp.getString("triggerMode", "BOTH") ?: "BOTH"
        set(v) = sp.edit().putString("triggerMode", v).apply()

    /** Photo-eye position as a fraction of the analysis frame along the motion axis. */
    var lineFraction: Float
        get() = sp.getFloat("lineFraction", 0.5f)
        set(v) = sp.edit().putFloat("lineFraction", v.coerceIn(0.2f, 0.8f)).apply()

    /**
     * How parts move AS SEEN ON SCREEN: "H" (left↔right, default) or "V" (up↕down). The photo-eye line is perpendicular to
     * it; the sensor axis is resolved per frame from its rotation (portrait camera frames are rotated 90°, clips may not be).
     */
    /** Camera zoom (1 = full view). Crops the view for preview, analysis and clips alike; shared by all camera screens. */
    var zoomRatio: Float
        get() = sp.getFloat("zoomRatio", 1f)
        set(v) = sp.edit().putFloat("zoomRatio", v.coerceIn(1f, 10f)).apply()

    var motion: String
        get() = sp.getString("motion", "H") ?: "H"
        set(v) = sp.edit().putString("motion", v).apply()

    /** Backbone choice id ("r18" default, "dinov2" once its model is pushed). Changing it requires re-teaching / rebuilding Twins. */
    var backbone: String
        get() = sp.getString("backbone", "r18") ?: "r18"
        set(v) = sp.edit().putString("backbone", v).apply()

    var vlmAuto: Boolean
        get() = sp.getBoolean("vlmAuto", true)
        set(v) = sp.edit().putBoolean("vlmAuto", v).apply()

    var voting: Boolean
        get() = sp.getBoolean("voting", true)
        set(v) = sp.edit().putBoolean("voting", v).apply()

    var muted: Boolean
        get() = sp.getBoolean("muted", false)
        set(v) = sp.edit().putBoolean("muted", v).apply()

    var cameraPreset: CameraPreset
        get() = sp.getString("cameraPreset", null)?.let { presetFromJson(JSONObject(it)) } ?: CameraPreset()
        set(p) = sp.edit().putString("cameraPreset", presetToJson(p).toString()).apply()

    companion object {
        fun presetToJson(p: CameraPreset) = JSONObject().apply {
            put("fps", p.fps); put("manualExposure", p.manualExposure); put("exposureNs", p.exposureNs); put("iso", p.iso)
            put("aeLock", p.aeLock); put("awbLock", p.awbLock); put("torch", p.torch)
            p.focusDiopters?.let { put("focusDiopters", it.toDouble()) }
            put("analysisWidth", p.analysisWidth); put("analysisHeight", p.analysisHeight)
        }

        fun presetFromJson(o: JSONObject) = CameraPreset(
            fps = o.optInt("fps", 30),
            manualExposure = o.optBoolean("manualExposure", false),
            exposureNs = o.optLong("exposureNs", 1_000_000L),
            iso = o.optInt("iso", 800),
            aeLock = o.optBoolean("aeLock", false),
            awbLock = o.optBoolean("awbLock", false),
            torch = false, // flash is always off (older saved presets had torch=true)
            focusDiopters = if (o.has("focusDiopters")) o.getDouble("focusDiopters").toFloat() else null,
            analysisWidth = o.optInt("analysisWidth", 1280),
            analysisHeight = o.optInt("analysisHeight", 720),
        )
    }
}
