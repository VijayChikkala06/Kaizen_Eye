package com.kaizeneye.v2.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun ScreenHeader(title: String, onBack: (() -> Unit)?, trailing: @Composable RowScope.() -> Unit = {}) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(Kz.Surface)
            .padding(horizontal = 8.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (onBack != null) TextButton(onClick = onBack) { Text("‹ Back", color = Kz.TextDim) }
        Text(title, fontSize = 18.sp, fontWeight = FontWeight.SemiBold, color = Kz.Text, modifier = Modifier.padding(start = 6.dp).weight(1f))
        trailing()
    }
}

@Composable
fun Card(modifier: Modifier = Modifier, border: Color = Kz.Line, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier
            .fillMaxWidth()
            .background(Kz.Surface, RoundedCornerShape(10.dp))
            .border(BorderStroke(1.dp, border), RoundedCornerShape(10.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
        content = content,
    )
}

@Composable
fun PrimaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, color: Color = Kz.Accent, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(52.dp),
        shape = RoundedCornerShape(10.dp),
        colors = ButtonDefaults.buttonColors(containerColor = color, contentColor = Color.White),
    ) { Text(text, fontSize = 16.sp, fontWeight = FontWeight.SemiBold) }
}

@Composable
fun SecondaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.height(46.dp),
        shape = RoundedCornerShape(10.dp),
        border = BorderStroke(1.dp, Kz.Line),
    ) { Text(text, color = Kz.Text, fontSize = 14.sp) }
}

/** Small status chip, e.g. an accelerator badge or "AIRPLANE MODE ✓". */
@Composable
fun Chip(text: String, color: Color = Kz.TextDim, modifier: Modifier = Modifier) {
    Box(
        modifier
            .border(1.dp, color.copy(alpha = 0.7f), RoundedCornerShape(6.dp))
            .padding(horizontal = 7.dp, vertical = 3.dp),
    ) { Text(text, color = color, fontSize = 11.sp, fontFamily = Kz.Mono) }
}

@Composable
fun KeyValue(key: String, value: String, valueColor: Color = Kz.Text) {
    Row(Modifier.fillMaxWidth()) {
        Text(key, color = Kz.TextDim, fontSize = 13.sp, modifier = Modifier.width(150.dp))
        Text(value, color = valueColor, fontSize = 13.sp, fontFamily = Kz.Mono)
    }
}

@Composable
fun Gap(h: Int = 8) = Spacer(Modifier.height(h.dp))

@Composable
fun Note(text: String, color: Color = Kz.TextDim) = Text(text, color = color, fontSize = 12.sp)
