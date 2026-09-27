package com.kaizeneye.v2.alarm

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log

/**
 * Andon-style reject alert (plan HMI): tone + vibration on REJECTS ONLY (defect / not the enrolled part), escalating after
 * three rejects in a row. Pass verdicts are silent. [muted] silences the tone but keeps the vibration.
 */
class Alarm(context: Context) {
    private val tone: ToneGenerator? = try {
        // The alarm stream: not silenced by the ring / notification switch or Do Not Disturb (a reject must be heard).
        ToneGenerator(AudioManager.STREAM_ALARM, 100)
    } catch (t: Throwable) {
        Log.w("KaizenAlarm", "no tone generator", t)
        null
    }
    private val vibrator: Vibrator? = try {
        (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
    } catch (t: Throwable) {
        null
    }

    @Volatile var muted = false

    /** [consecutive] = rejects in a row including this one. */
    fun reject(consecutive: Int) {
        val escalate = consecutive >= 3
        try {
            if (!muted) {
                tone?.startTone(
                    if (escalate) ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK else ToneGenerator.TONE_CDMA_ABBR_ALERT,
                    if (escalate) 450 else 160,
                )
            }
        } catch (_: Throwable) {
        }
        try {
            val effect = if (escalate) VibrationEffect.createWaveform(longArrayOf(0, 160, 80, 160, 80, 320), -1)
            else VibrationEffect.createOneShot(140, VibrationEffect.DEFAULT_AMPLITUDE)
            vibrator?.vibrate(effect)
        } catch (_: Throwable) {
        }
    }

    /** Short neutral click (e.g. a calibration sample was accepted). */
    fun tick() {
        try {
            if (!muted) tone?.startTone(ToneGenerator.TONE_PROP_BEEP, 60)
        } catch (_: Throwable) {
        }
    }

    fun release() {
        try {
            tone?.release()
        } catch (_: Throwable) {
        }
    }
}
