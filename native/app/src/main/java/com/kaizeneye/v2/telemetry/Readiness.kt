package com.kaizeneye.v2.telemetry

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.provider.Settings
import androidx.core.content.ContextCompat

/** Offline / permission proof for the readiness panel and the self-test (plan: "airplane-mode proof", no INTERNET permission). */
object Readiness {
    fun airplaneMode(ctx: Context): Boolean =
        Settings.Global.getInt(ctx.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1

    /** Every permission in the installed (merged) manifest. */
    fun requestedPermissions(ctx: Context): List<String> = try {
        ctx.packageManager.getPackageInfo(ctx.packageName, PackageManager.GET_PERMISSIONS).requestedPermissions?.toList() ?: emptyList()
    } catch (t: Throwable) {
        emptyList()
    }

    fun internetDeclared(ctx: Context): Boolean = requestedPermissions(ctx).any {
        it == Manifest.permission.INTERNET || it == Manifest.permission.ACCESS_NETWORK_STATE
    }

    fun cameraGranted(ctx: Context): Boolean =
        ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    /** Permissions beyond the allow-list (CAMERA, VIBRATE, AndroidX's own signature permission). Empty = clean. */
    fun unexpectedPermissions(ctx: Context): List<String> = requestedPermissions(ctx).filterNot {
        it == Manifest.permission.CAMERA || it == Manifest.permission.VIBRATE || it.endsWith(".DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION")
    }
}
