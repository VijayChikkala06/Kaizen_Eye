package com.kaizeneye.v2

import android.content.Context
import com.kaizeneye.runtime.VlmClient
import com.kaizeneye.v2.alarm.Alarm
import com.kaizeneye.v2.camera.CameraController
import com.kaizeneye.v2.ml.EngineHolder
import com.kaizeneye.v2.pipeline.PipelineHub
import com.kaizeneye.v2.telemetry.JsonlLog
import com.kaizeneye.v2.telemetry.SoakRecorder
import com.kaizeneye.v2.telemetry.ThermalMonitor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

/**
 * Main-process service locator (created lazily from the Activity — Application.onCreate stays empty because it also runs in
 * the :vlm and :probe processes, which must not touch the camera or LiteRT vision stack).
 */
class AppGraph private constructor(val context: Context) {
    val scope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default +
            kotlinx.coroutines.CoroutineExceptionHandler { _, t -> android.util.Log.e("KaizenApp", "background job failed", t) },
    )
    /** Camera permission as last observed (the preview re-binds when it is granted). */
    val cameraGranted = kotlinx.coroutines.flow.MutableStateFlow(com.kaizeneye.v2.telemetry.Readiness.cameraGranted(context))
    val dirs = AppDirs(context).ensure()
    val prefs = Prefs(context)
    val log = JsonlLog(dirs.logs)
    val thermal = ThermalMonitor(context)
    val alarm = Alarm(context)
    val camera = CameraController(context)
    val engines = EngineHolder(context, scope)
    val vlm = VlmClient(context)
    val hub = PipelineHub(this)
    val soak = SoakRecorder(this)

    companion object {
        @Volatile private var instance: AppGraph? = null

        fun get(ctx: Context): AppGraph = instance ?: synchronized(this) {
            instance ?: AppGraph(ctx.applicationContext).also { instance = it }
        }
    }
}
