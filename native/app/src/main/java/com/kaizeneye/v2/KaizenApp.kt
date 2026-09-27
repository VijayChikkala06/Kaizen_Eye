package com.kaizeneye.v2

import android.app.Application

/**
 * Deliberately empty: Application.onCreate runs in EVERY process of the app (main, :vlm, :probe). The VLM process must never
 * load the vision LiteRT (liblitertlm_jni.so embeds its own LiteRT), so all main-process setup happens lazily in [AppGraph].
 */
class KaizenApp : Application()
