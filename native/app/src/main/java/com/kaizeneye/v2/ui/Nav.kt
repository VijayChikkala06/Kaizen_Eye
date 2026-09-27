package com.kaizeneye.v2.ui

import androidx.compose.runtime.mutableStateListOf
import java.io.File

/** App screens. A tiny back stack instead of a navigation library (fewer dependencies, predictable back behaviour). */
sealed interface Screen {
    data object Home : Screen
    data object Teach : Screen
    data class Inspect(val mode: LineMode, val replayClip: File? = null, val paced: Boolean = true) : Screen
    data object Certificate : Screen
    data object Telemetry : Screen
    data object Readiness : Screen
    data object Clips : Screen
    data object Negatives : Screen
    data class SelfTest(val autoStart: String? = null) : Screen
}

enum class LineMode { INSPECT, CALIBRATE }

class Navigator(start: Screen = Screen.Home) {
    val stack = mutableStateListOf(start)
    val current: Screen get() = stack.last()
    fun go(s: Screen) {
        stack.add(s)
    }

    fun replace(s: Screen) {
        stack[stack.lastIndex] = s
    }

    fun back(): Boolean {
        if (stack.size <= 1) return false
        stack.removeAt(stack.lastIndex)
        return true
    }

    fun home() {
        while (stack.size > 1) stack.removeAt(stack.lastIndex)
    }
}
