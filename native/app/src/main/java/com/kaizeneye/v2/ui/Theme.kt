package com.kaizeneye.v2.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * HMI palette (plan "HMI"): a grey UI where colour is reserved for abnormal or state-changing events — thin grey while
 * tracking, amber while judging, a brief green pass, thick red for a defect, violet for "not the enrolled part".
 */
object Kz {
    val Bg = Color(0xFF101418)
    val Surface = Color(0xFF1A2027)
    val SurfaceHi = Color(0xFF232B34)
    val Line = Color(0xFF3A4550)
    val Text = Color(0xFFE8EEF4)
    val TextDim = Color(0xFF9AA8B6)
    val Tracking = Color(0xFFB8C4CF)
    val Judging = Color(0xFFF2A900)
    val Pass = Color(0xFF2EB872)
    val Defect = Color(0xFFE5484D)
    val NotEnrolled = Color(0xFF9B7BFF)
    val Reframe = Color(0xFF7D8A96)
    val Accent = Color(0xFF4A9EFF)
    val Warn = Color(0xFFF2A900)
    val Mono = FontFamily.Monospace
}

private val scheme = darkColorScheme(
    primary = Kz.Accent,
    onPrimary = Color.White,
    secondary = Kz.Tracking,
    background = Kz.Bg,
    onBackground = Kz.Text,
    surface = Kz.Surface,
    onSurface = Kz.Text,
    surfaceVariant = Kz.SurfaceHi,
    onSurfaceVariant = Kz.TextDim,
    error = Kz.Defect,
    outline = Kz.Line,
)

private val type = Typography(
    titleLarge = TextStyle(fontSize = 22.sp, fontWeight = FontWeight.SemiBold, color = Kz.Text),
    titleMedium = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = Kz.Text),
    bodyLarge = TextStyle(fontSize = 16.sp, color = Kz.Text),
    bodyMedium = TextStyle(fontSize = 14.sp, color = Kz.Text),
    bodySmall = TextStyle(fontSize = 12.sp, color = Kz.TextDim),
    labelLarge = TextStyle(fontSize = 15.sp, fontWeight = FontWeight.SemiBold),
    labelSmall = TextStyle(fontSize = 11.sp, color = Kz.TextDim),
)

@Composable
fun KaizenTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = scheme, typography = type, content = content)
}
