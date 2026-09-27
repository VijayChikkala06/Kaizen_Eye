package com.kaizeneye.v2

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import android.provider.Settings
import android.system.Os
import android.system.OsConstants
import android.util.Range
import android.util.Size
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** One-shot snapshot of everything the plan needs to know about a device: SoC strings, camera limits, thermal/battery APIs, DSP hints. */
object DevicePassport {

    fun collect(ctx: Context): JSONObject = JSONObject().apply {
        put("collectedAtMs", System.currentTimeMillis())
        put("build", build())
        put("system", system(ctx))
        put("props", props())
        put("cameras", cameras(ctx))
        put("thermal", thermal(ctx))
        put("battery", battery(ctx))
        put("npuHints", npuHints())
    }

    /** Writes to the app-specific external dir (also proves that dir exists for `adb push`). */
    fun save(ctx: Context, passport: JSONObject): File? {
        val dir = ctx.getExternalFilesDir(null) ?: return null
        val f = File(dir, "device-passport.json")
        f.writeText(passport.toString(2))
        return f
    }

    private fun build() = JSONObject().apply {
        put("manufacturer", Build.MANUFACTURER)
        put("brand", Build.BRAND)
        put("model", Build.MODEL)
        put("device", Build.DEVICE)
        put("product", Build.PRODUCT)
        put("hardware", Build.HARDWARE)
        put("board", Build.BOARD)
        put("socManufacturer", Build.SOC_MANUFACTURER)
        put("socModel", Build.SOC_MODEL)
        put("sdkInt", Build.VERSION.SDK_INT)
        put("release", Build.VERSION.RELEASE)
        put("securityPatch", Build.VERSION.SECURITY_PATCH)
        put("fingerprint", Build.FINGERPRINT)
        put("display", Build.DISPLAY)
        put("supportedAbis", JSONArray(Build.SUPPORTED_ABIS.toList()))
    }

    private fun system(ctx: Context) = JSONObject().apply {
        put("pageSizeBytes", Os.sysconf(OsConstants._SC_PAGESIZE))
        put("nativeLibraryDir", ctx.applicationInfo.nativeLibraryDir)
        put("filesDir", ctx.filesDir.absolutePath)
        val ext = ctx.getExternalFilesDir(null)
        put("externalFilesDir", ext?.absolutePath ?: "null")
        if (ext != null) put("externalFreeMB", StatFs(ext.absolutePath).availableBytes / (1024L * 1024L))
        put("airplaneModeOn", Settings.Global.getInt(ctx.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1)
        val pm = ctx.packageManager
        put("features", JSONObject().apply {
            for (name in listOf(
                "android.hardware.camera.flash", "android.hardware.camera.level.full",
                "android.hardware.camera.capability.manual_sensor", "android.hardware.camera.capability.raw",
                "android.hardware.npu", "android.hardware.vulkan.version", "android.hardware.sensor.gyroscope",
            )) put(name, pm.hasSystemFeature(name))
        })
        put("osArch", System.getProperty("os.arch") ?: "")
    }

    private fun getprop(key: String): String = try {
        val p = Runtime.getRuntime().exec(arrayOf("getprop", key))
        p.inputStream.bufferedReader().use { it.readText().trim() }
    } catch (e: Exception) {
        "err:${e.javaClass.simpleName}"
    }

    private fun props() = JSONObject().apply {
        for (k in listOf(
            "ro.soc.model", "ro.soc.manufacturer", "ro.board.platform", "ro.hardware", "ro.hardware.chipname",
            "ro.product.board", "ro.product.vendor.model", "ro.vivo.product.model", "ro.build.display.id",
            "ro.build.version.sdk", "ro.build.version.release",
        )) put(k, getprop(k))
    }

    private fun rangeToJson(r: Range<*>?): JSONArray? = r?.let { JSONArray(listOf<Any>(it.lower, it.upper)) }

    private fun cameras(ctx: Context): JSONArray {
        val out = JSONArray()
        val cm = ctx.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        for (id in cm.cameraIdList) {
            val c = cm.getCameraCharacteristics(id)
            val o = JSONObject()
            o.put("id", id)
            o.put("lensFacing", c.get(CameraCharacteristics.LENS_FACING))
            o.put("hardwareLevel", c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL))
            o.put("capabilities", JSONArray(c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toList() ?: emptyList<Int>()))
            o.put("exposureTimeNs", rangeToJson(c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)))
            o.put("isoRange", rangeToJson(c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)))
            o.put(
                "aeFpsRanges",
                JSONArray((c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: emptyArray<Range<Int>>()).map { "${it.lower}-${it.upper}" }),
            )
            o.put("flashAvailable", c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE))
            o.put("pixelArray", c.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)?.toString())
            o.put("physicalSizeMm", c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.toString())
            o.put("focalLengthsMm", JSONArray(c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.toList() ?: emptyList<Float>()))
            o.put("minFocusDistanceDiopters", c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE))
            o.put("antibandingModes", JSONArray(c.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_ANTIBANDING_MODES)?.toList() ?: emptyList<Int>()))
            o.put("aeLockAvailable", c.get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE))
            o.put("awbLockAvailable", c.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE))
            o.put("sensorOrientation", c.get(CameraCharacteristics.SENSOR_ORIENTATION))
            o.put("physicalIds", JSONArray(c.physicalCameraIds.toList()))
            val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            o.put("yuvSizes", JSONArray((map?.getOutputSizes(ImageFormat.YUV_420_888) ?: emptyArray<Size>()).take(14).map { "${it.width}x${it.height}" }))
            out.put(o)
        }
        return out
    }

    private fun thermal(ctx: Context): JSONObject {
        val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
        return JSONObject().apply {
            put("status", pm.currentThermalStatus)
            val h = try { pm.getThermalHeadroom(10) } catch (e: Exception) { Float.NaN }
            put("headroom10s", if (h.isNaN()) "NaN" else h.toString())
            put("sustainedPerformanceSupported", pm.isSustainedPerformanceModeSupported)
        }
    }

    private fun battery(ctx: Context): JSONObject {
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val i: Intent? = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return JSONObject().apply {
            put("capacityPct", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY))
            put("currentNowUa", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW))
            put("currentAvgUa", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE))
            put("chargeCounterUah", bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER))
            put("energyCounterNwh", bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER))
            put("tempTenthsC", i?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE)
            put("voltageMv", i?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, Int.MIN_VALUE) ?: Int.MIN_VALUE)
            put("plugged", i?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1)
            put("status", i?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1)
        }
    }

    private fun npuHints() = JSONObject().apply {
        for (p in listOf(
            "/vendor/lib64/libcdsprpc.so", "/system/lib64/libcdsprpc.so", "/system/vendor/lib64/libcdsprpc.so",
            "/vendor/lib/rfsa/adsp", "/vendor/dsp", "/odm/lib/rfsa/adsp", "/dsp",
        )) {
            put(p, JSONObject().apply {
                val f = File(p)
                put("exists", f.exists())
                put("isDir", f.isDirectory)
                if (f.isDirectory) {
                    val names = try { f.list()?.toList()?.take(40) } catch (e: Exception) { null }
                    put("entries", JSONArray(names ?: emptyList<String>()))
                }
            })
        }
    }
}
