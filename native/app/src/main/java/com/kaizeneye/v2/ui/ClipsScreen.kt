package com.kaizeneye.v2.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.kaizeneye.v2.AppGraph
import com.kaizeneye.v2.camera.CameraController
import com.kaizeneye.v2.camera.ReplayFrameSource
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Labelled clip recording (same camera preset) + REPLAY of any clip through the identical pipeline. */
@Composable
fun ClipsScreen(g: AppGraph, nav: Navigator) {
    val labels = listOf("teach", "good", "defect", "wrong", "rotated", "lookalike", "calib", "negatives", "demo")
    var label by remember { mutableStateOf("good") }
    var recorderOpen by remember { mutableStateOf(false) }
    var refresh by remember { mutableIntStateOf(0) }
    var message by remember { mutableStateOf<String?>(null) }
    var exporting by remember { mutableStateOf(false) }
    val stats by g.camera.stats.collectAsStateWithLifecycle()
    val clips = remember(refresh) { listClips(g) }
    var confirmDelete by remember { mutableStateOf<File?>(null) }
    confirmDelete?.let { f ->
        androidx.compose.material3.AlertDialog(
            onDismissRequest = { confirmDelete = null },
            title = { Text("Delete ${f.name}?") },
            confirmButton = { TextButton(onClick = { f.deleteRecursively(); refresh++; confirmDelete = null }) { Text("Delete", color = Kz.Defect) } },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Keep") } },
        )
    }

    Column(Modifier.fillMaxSize()) {
        ScreenHeader("Clips & replay", onBack = { nav.back() })
        Row(Modifier.horizontalScroll(rememberScrollState()).padding(8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            labels.forEach { l -> FilterChip(selected = l == label, onClick = { label = l }, label = { Text(l) }) }
        }
        if (recorderOpen) {
            CameraPreview(g, CameraController.Mode.RECORD, Modifier.fillMaxWidth().height(320.dp), routeFrames = false)
            Row(Modifier.padding(8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                if (!stats.recording) {
                    PrimaryButton("● REC '$label'", Modifier.weight(1f), color = Kz.Defect) {
                        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
                        val f = File(g.dirs.clips, "${label}_$stamp.mp4")
                        val ok = g.camera.startRecording(f) { file, err ->
                            message = err ?: "saved ${file?.name}"
                            refresh++
                        }
                        if (!ok) message = "recorder not ready"
                    }
                } else {
                    PrimaryButton("■ STOP", Modifier.weight(1f)) { g.camera.stopRecording() }
                }
                SecondaryButton("Close") { recorderOpen = false }
            }
        } else {
            Row(Modifier.padding(horizontal = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PrimaryButton("RECORD A '$label' CLIP", Modifier.weight(1f)) { recorderOpen = true }
            }
            androidx.compose.foundation.layout.Box(Modifier.padding(horizontal = 12.dp, vertical = 4.dp)) {
                Note("Label = what the clip shows (good parts, seeded defects, wrong objects…). Start every clip with ½ s of EMPTY sheet. Replay feeds it through the same pipeline.", Kz.TextDim)
            }
            Row(Modifier.padding(horizontal = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SecondaryButton(if (exporting) "EXPORTING…" else "Export eval set (teach + labelled clips)", Modifier.weight(1f), enabled = !exporting) {
                    exporting = true
                    g.scope.launch {
                        message = try {
                            val dir = com.kaizeneye.v2.export.EvalBatch(g).run { message = it }
                            "export ready: ${dir.absolutePath}"
                        } catch (t: Throwable) {
                            "export failed: ${t.message}"
                        }
                        exporting = false
                    }
                }
            }
        }
        message?.let { Text(it, color = Kz.Accent, fontSize = 12.sp, modifier = Modifier.padding(8.dp)) }
        LazyColumn(Modifier.fillMaxSize().padding(horizontal = 8.dp)) {
            items(clips, key = { it.absolutePath }) { f ->
                Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(f.name, color = Kz.Text, fontSize = 14.sp)
                        Note(if (f.isDirectory) "JPEG sequence" else "${f.length() / 1_000_000} MB mp4")
                    }
                    TextButton(onClick = {
                        message = "teaching from ${f.name}…"
                        g.scope.launch { message = g.hub.teachFromClip(f) }
                    }) { Text("Teach") }
                    TextButton(onClick = { nav.go(Screen.Inspect(LineMode.INSPECT, replayClip = f, paced = true)) }) { Text("Replay") }
                    TextButton(onClick = { nav.go(Screen.Inspect(LineMode.INSPECT, replayClip = f, paced = false)) }) { Text("Fast") }
                    TextButton(onClick = { confirmDelete = f }) { Text("Del", color = Kz.Defect) }
                }
            }
        }
    }
}

private fun listClips(g: AppGraph): List<File> =
    (g.dirs.clips.listFiles()?.toList() ?: emptyList()).filter { ReplayFrameSource.isClip(it) }.sortedByDescending { it.lastModified() }
