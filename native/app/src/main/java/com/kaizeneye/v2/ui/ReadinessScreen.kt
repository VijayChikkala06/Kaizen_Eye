package com.kaizeneye.v2.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.runtime.LocatedModel
import com.kaizeneye.runtime.ModelAsset
import com.kaizeneye.runtime.ModelLocator
import com.kaizeneye.runtime.VlmState
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.DevicePassport
import com.kaizeneye.v2.ml.Models
import com.kaizeneye.v2.telemetry.Readiness
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Demo readiness: offline proof, permissions, model files + SHA-256, accelerator decision, VLM, passport, imports/exports. */
@Composable
fun ReadinessScreen(g: AppGraph, nav: Navigator) {
    val ctx = g.context
    val scope = rememberCoroutineScope()
    val vlm by g.vlm.state.collectAsStateWithLifecycle()
    var models by remember { mutableStateOf<List<Pair<ModelAsset, LocatedModel?>>>(emptyList()) }
    var checking by remember { mutableStateOf(false) }
    var message by remember { mutableStateOf<String?>(null) }
    var importTarget by remember { mutableStateOf<ModelAsset?>(null) }
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        val target = importTarget
        if (uri != null && target != null) scope.launch {
            message = "importing ${target.fileName}…"
            message = try {
                val r = withContext(Dispatchers.IO) { ModelLocator.importFromUri(ctx, uri, target) }
                "imported ${target.fileName}: sha ${if (r.sha256Ok) "OK" else "MISMATCH"}"
            } catch (t: Throwable) {
                "import failed: ${t.message}"
            }
        }
    }
    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Readiness", onBack = { nav.back() })
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Card {
                Text("Offline proof", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                val airplane = Readiness.airplaneMode(ctx)
                KeyValue("airplane mode", if (airplane) "ON ✓" else "off", if (airplane) Kz.Pass else Kz.Warn)
                val net = Readiness.internetDeclared(ctx)
                KeyValue("INTERNET permission", if (net) "DECLARED ✗" else "absent ✓", if (net) Kz.Defect else Kz.Pass)
                val extra = Readiness.unexpectedPermissions(ctx)
                KeyValue("other permissions", if (extra.isEmpty()) "none ✓ (CAMERA, VIBRATE only)" else extra.joinToString(), if (extra.isEmpty()) Kz.Pass else Kz.Defect)
                KeyValue("camera granted", if (Readiness.cameraGranted(ctx)) "yes" else "NO", if (Readiness.cameraGranted(ctx)) Kz.Text else Kz.Defect)
                Note("The APK has no INTERNET permission at all (enforced by tools/apk-audit.ps1), so no model, image or result can leave the phone.")
            }
            Card {
                Text("Models (SHA-256 pinned)", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                if (models.isEmpty()) Note("Tap CHECK to locate every model and verify its SHA-256 (the 942 MB VLM takes a few seconds once; then cached).")
                for ((a, l) in models) {
                    Text(
                        "${a.fileName}\n   ${l?.let { "${it.source} · ${it.bytes / 1_000_000} MB · sha ${if (it.sha256Ok) "OK" else "MISMATCH"}" } ?: "not found"}",
                        color = if (l?.sha256Ok == true) Kz.Text else if (l == null) Kz.TextDim else Kz.Defect, fontSize = 12.sp, fontFamily = Kz.Mono,
                    )
                }
                PrimaryButton(if (checking) "CHECKING…" else "CHECK MODELS", enabled = !checking) {
                    checking = true
                    scope.launch {
                        models = withContext(Dispatchers.IO) { Models.all.map { it to runCatching { ModelLocator.locate(ctx, it) }.getOrNull() } }
                        checking = false
                    }
                }
                SecondaryButton("Import FastVLM (.litertlm) via file picker") { importTarget = Models.FASTVLM; importer.launch(arrayOf("*/*")) }
                Note("Or: adb push FastVLM-0.5B.qualcomm.sm8850.litertlm /sdcard/Android/data/com.kaizeneye.v2/files/models/")
            }
            Card {
                Text("Backbone", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                val current = Models.choice(g.prefs.backbone)
                KeyValue("in use", "${current.label} · ${g.engines.badge()}")
                Note("Twins remember the backbone that built them (fingerprint); switching makes other Twins unloadable until they are re-taught. DINOv2 needs dinov2_s14_448_fp16w.tflite pushed to …/files/models/.")
                androidx.compose.foundation.layout.Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    for (c in Models.choices + (if (com.kaizeneye.v2.BuildConfig.DEBUG) listOf(Models.DEBUG_CHOICE) else emptyList())) {
                        SecondaryButton(c.label, Modifier.weight(1f), enabled = c.id != current.id) {
                            g.prefs.backbone = c.id
                            g.engines.load(c)
                            message = "loading ${c.label}… (re-teach Twins for this backbone)"
                        }
                    }
                }
            }
            Card {
                Text("Explanations (offline VLM, separate process)", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                KeyValue(
                    "state",
                    when (val v = vlm) {
                        VlmState.Idle -> "idle"
                        VlmState.Loading -> "loading…"
                        is VlmState.Ready -> "ready on ${v.backend} (init ${v.initMs} ms)"
                        is VlmState.Failed -> "unavailable: ${v.message} → template explanations"
                    },
                )
                SecondaryButton("Start / warm VLM") { g.vlm.bind() }
            }
            Card {
                Text("Device passport & exports", fontWeight = FontWeight.SemiBold, color = Kz.Text)
                SecondaryButton("Write device passport") {
                    scope.launch {
                        message = withContext(Dispatchers.IO) { DevicePassport.save(ctx, DevicePassport.collect(ctx))?.absolutePath ?: "external files dir unavailable" }
                    }
                }
                SecondaryButton("Export active Twin") {
                    scope.launch {
                        message = withContext(Dispatchers.IO) { g.hub.active.value?.let { g.hub.exportTwin(it.id)?.absolutePath } ?: "no active Twin" }
                    }
                }
                Note("Logs: ${g.dirs.logs.absolutePath}")
                Note("Clips: ${g.dirs.clips.absolutePath}")
            }
            message?.let { Text(it, color = Kz.Accent, fontSize = 12.sp, fontFamily = Kz.Mono) }
        }
    }
}
