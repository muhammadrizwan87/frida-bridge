/*
 * app-detection.c
 *
 * Application detection and file directory resolution via JNI reflection.
 *
 * This module handles the complex task of finding the correct Android
 * Application context in multi-process container environments. It
 * employs multiple resolution paths and strictly accepts only applications
 * containing the expected payload marker, preventing premature acceptance
 * of proxy container applications.
 */

#include <jni.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <linux/limits.h>

#include "frida-bridge.h"
#include "debug-logging.h"
#include "jni-helpers.h"

extern int is_target_app_ready(const char *files_dir);

/* ═══════════════════════════════════════════════════════════════════════════
 * Application Resolution — Multiple Framework Entry Points
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * resolve_application_jni - Resolve a live Application jobject using
 * multiple framework entry points.
 *
 * This function attempts to obtain the current Application object via
 * several reflection paths, improving reliability during early process
 * startup and in container environments where one path may be unavailable
 * while others work.
 *
 * Resolution order:
 *   1. ActivityThread.currentApplication()
 *      Primary entry point, returns the static singleton if available.
 *   2. ActivityThread.currentActivityThread().getApplication()
 *      Obtains the ActivityThread instance first, then calls instance
 *      method, may succeed when the static method returns null.
 *   3. AppGlobals.getInitialApplication()
 *      Last‑resort fallback; returns the initial application object.
 *
 * All JNI references are handled locally; on success returns a local
 * reference that the caller must delete.
 *
 * @env:     JNI environment pointer.
 * @cls_at:  Pre‑resolved ActivityThread class reference.
 * @return:  Local reference to Application, or NULL if all paths fail.
 */
static jobject resolve_application_jni(JNIEnv *env, jclass cls_at) {
  if (!env || !cls_at) return NULL;

  LOGD("resolve_application_jni: attempting multiple resolution paths");

  /* ───────────────────────────────────────────────────────────────────────── */
  /* PATH 1: ActivityThread.currentApplication() (Primary) */
  /* ───────────────────────────────────────────────────────────────────────── */
  {
    jmethodID mid_ca = (*env)->GetStaticMethodID(
        env, cls_at, "currentApplication", "()Landroid/app/Application;");
    if (mid_ca) {
      jobject app = (*env)->CallStaticObjectMethod(env, cls_at, mid_ca);
      JNI_CLEAR_EXCEPTION(env);
      if (app) {
        LOGD("resolve_application_jni: PATH 1 SUCCESS - currentApplication() returned object");
        return app;
      } else {
        LOGD("resolve_application_jni: PATH 1 returned NULL");
      }
    } else {
      JNI_CLEAR_EXCEPTION(env);
      LOGD("resolve_application_jni: PATH 1 method not found");
    }
  }

  /* ───────────────────────────────────────────────────────────────────────── */
  /* PATH 2: ActivityThread.currentActivityThread().getApplication() */
  /* ───────────────────────────────────────────────────────────────────────── */
  {
    jmethodID mid_cat = (*env)->GetStaticMethodID(
        env, cls_at, "currentActivityThread", "()Landroid/app/ActivityThread;");
    if (mid_cat) {
      jobject at = (*env)->CallStaticObjectMethod(env, cls_at, mid_cat);
      JNI_CLEAR_EXCEPTION(env);
      if (at) {
        LOGD("resolve_application_jni: PATH 2 got ActivityThread instance");
        
        /* Now call getApplication() on the instance */
        jmethodID mid_get_app = (*env)->GetMethodID(
            env, cls_at, "getApplication", "()Landroid/app/Application;");
        if (mid_get_app) {
          jobject app = (*env)->CallObjectMethod(env, at, mid_get_app);
          JNI_CLEAR_EXCEPTION(env);
          (*env)->DeleteLocalRef(env, at);
          
          if (app) {
            LOGD("resolve_application_jni: PATH 2 SUCCESS - "
                 "currentActivityThread().getApplication() returned object");
            return app;
          } else {
            LOGD("resolve_application_jni: PATH 2 getApplication() returned NULL");
          }
        } else {
          JNI_CLEAR_EXCEPTION(env);
          LOGD("resolve_application_jni: PATH 2 getApplication() method not found");
          (*env)->DeleteLocalRef(env, at);
        }
      } else {
        LOGD("resolve_application_jni: PATH 2 currentActivityThread() returned NULL");
      }
    } else {
      JNI_CLEAR_EXCEPTION(env);
      LOGD("resolve_application_jni: PATH 2 currentActivityThread() method not found");
    }
  }

  /* ───────────────────────────────────────────────────────────────────────── */
  /* PATH 3: AppGlobals.getInitialApplication() (Last Resort) */
  /* ───────────────────────────────────────────────────────────────────────── */
  {
    jclass cls_ag = (*env)->FindClass(env, "android/app/AppGlobals");
    if (cls_ag) {
      jmethodID mid_gia = (*env)->GetStaticMethodID(
          env, cls_ag, "getInitialApplication", "()Landroid/app/Application;");
      if (mid_gia) {
        jobject app = (*env)->CallStaticObjectMethod(env, cls_ag, mid_gia);
        JNI_CLEAR_EXCEPTION(env);
        (*env)->DeleteLocalRef(env, cls_ag);
        
        if (app) {
          LOGD("resolve_application_jni: PATH 3 SUCCESS - "
               "AppGlobals.getInitialApplication() returned object");
          return app;
        } else {
          LOGD("resolve_application_jni: PATH 3 getInitialApplication() returned NULL");
        }
      } else {
        JNI_CLEAR_EXCEPTION(env);
        LOGD("resolve_application_jni: PATH 3 method not found");
        (*env)->DeleteLocalRef(env, cls_ag);
      }
    } else {
      JNI_CLEAR_EXCEPTION(env);
      LOGD("resolve_application_jni: PATH 3 AppGlobals class not found");
    }
  }

  LOGW("resolve_application_jni: all resolution paths exhausted, returning NULL");
  return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * JNI File Directory Extraction
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * extract_files_dir_from_app - Extract getFilesDir().getAbsolutePath() from jobject.
 *
 * Calls Context.getFilesDir().getAbsolutePath() via JNI reflection on a
 * given Application/Context jobject.
 *
 * This function has no retry logic—it attempts a single extraction and
 * succeeds or fails. Used both for payload probing and final result extraction.
 *
 * All JNI references are carefully managed:
 * - Local refs are deleted in error paths and at cleanup.
 * - String refs are released before deletion.
 * - No orphaned or stale references are left behind.
 *
 * @env:      JNI environment pointer.
 * @app:      Application/Context jobject to query.
 * @out_buf:  Output buffer for the absolute path.
 * @out_sz:   Size of @out_buf.
 * @return:   1 on success with path in @out_buf, 0 on failure (out_buf cleared).
 */
static int extract_files_dir_from_app(JNIEnv *env, jobject app,
                                       char *out_buf, size_t out_sz) {
  if (!env || !app || !out_buf || !out_sz) return 0;

  out_buf[0] = '\0';

  /* Find android.content.Context class. */
  jclass cls_ctx = (*env)->FindClass(env, "android/content/Context");
  if (!cls_ctx) {
    JNI_CLEAR_EXCEPTION(env);
    return 0;
  }

  /* Get getFilesDir() method ID. */
  jmethodID mid_gfd = (*env)->GetMethodID(env, cls_ctx, "getFilesDir", 
                                           "()Ljava/io/File;");
  if (!mid_gfd) {
    JNI_CLEAR_EXCEPTION(env);
    (*env)->DeleteLocalRef(env, cls_ctx);
    return 0;
  }

  /* Call app.getFilesDir() to get File object. */
  jobject file = (*env)->CallObjectMethod(env, app, mid_gfd);
  JNI_CLEAR_EXCEPTION(env);
  if (!file) {
    (*env)->DeleteLocalRef(env, cls_ctx);
    return 0;
  }

  /* Find java.io.File class. */
  jclass cls_file = (*env)->FindClass(env, "java/io/File");
  if (!cls_file) {
    JNI_CLEAR_EXCEPTION(env);
    (*env)->DeleteLocalRef(env, file);
    (*env)->DeleteLocalRef(env, cls_ctx);
    return 0;
  }

  /* Get getAbsolutePath() method ID. */
  jmethodID mid_gap = (*env)->GetMethodID(env, cls_file, "getAbsolutePath", 
                                           "()Ljava/lang/String;");
  if (!mid_gap) {
    JNI_CLEAR_EXCEPTION(env);
    (*env)->DeleteLocalRef(env, cls_file);
    (*env)->DeleteLocalRef(env, file);
    (*env)->DeleteLocalRef(env, cls_ctx);
    return 0;
  }

  /* Call file.getAbsolutePath() to get path string. */
  jstring path_str = (jstring)(*env)->CallObjectMethod(env, file, mid_gap);
  JNI_CLEAR_EXCEPTION(env);

  int ok = 0;
  if (path_str) {
    const char *chars = (*env)->GetStringUTFChars(env, path_str, NULL);
    if (chars && chars[0]) {
      strncpy(out_buf, chars, out_sz - 1);
      out_buf[out_sz - 1] = '\0';
      ok = 1;
      LOGD("extract_files_dir_from_app: extracted path '%s'", out_buf);
    }
    if (chars)
      (*env)->ReleaseStringUTFChars(env, path_str, chars);
    (*env)->DeleteLocalRef(env, path_str);
  }

  /* Clean up all local references. */
  (*env)->DeleteLocalRef(env, cls_file);
  (*env)->DeleteLocalRef(env, file);
  (*env)->DeleteLocalRef(env, cls_ctx);

  return ok;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Application Polling with Payload Verification
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * get_files_dir_jni - Obtain the target application's files directory.
 *
 * Polls for an Application object containing the expected payload marker
 * for up to 30 seconds. Uses resolve_application_jni() for robust
 * resolution across multiple framework entry points.
 *
 * POLLING BEHAVIOR:
 * In container environments, a proxy application may appear before the
 * real target. To prevent premature acceptance, this function strictly
 * requires the payload marker to be present; it never returns a path
 * without it. The poll continues for the full timeout if necessary.
 *
 * @out_buf:  Output buffer for the files directory path.
 * @out_sz:   Size of @out_buf.
 * @return:   1 on success with path in @out_buf, 0 on failure.
 */
int get_files_dir_jni(char *out_buf, size_t out_sz) {
  if (!out_buf || !out_sz) return 0;

  /* Get the global JavaVM reference set by JNI_OnLoad. */
  JavaVM *jvm = g_jvm;
  if (!jvm) {
    LOGW("get_files_dir_jni: g_jvm is NULL, JavaVM not initialized");
    return 0;
  }

  JNIEnv *env = NULL;
  int attached = 0;
  int result = 0;

  /* Get or attach to the JNI environment. */
  jint rc = (*jvm)->GetEnv(jvm, (void **)&env, JNI_VERSION_1_6);
  if (rc == JNI_EDETACHED) {
    if ((*jvm)->AttachCurrentThread(jvm, &env, NULL) != JNI_OK) {
      LOGW("get_files_dir_jni: AttachCurrentThread failed");
      return 0;
    }
    attached = 1;
  } else if (rc != JNI_OK) {
    LOGW("get_files_dir_jni: GetEnv failed with code %d", rc);
    return 0;
  }

  jclass cls_at = NULL;
  jobject app = NULL;
  bool resolved_any_app = false;

  #define JNI_CLR() \
  do { \
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env); \
  } while (0)

  /* Find ActivityThread class. */
  cls_at = (*env)->FindClass(env, "android/app/ActivityThread");
  if (!cls_at) {
    JNI_CLR();
    goto jni_done;
  }

  /*
   * POLLING LOOP: Wait for an Application with the payload marker.
   *
   * For each iteration:
   *   1. Resolve an Application object via multiple framework paths.
   *   2. Extract its files directory path.
   *   3. Check for the payload marker (GADGET_SUBDIR).
   *   4. If found, accept and return the path.
   *   5. If not found, continue polling (never accept without payload).
   *
   * This ensures a proxy application is never mistaken for the target.
   */
  for (int i = 0; i < 30; ++i) {
    if (app) {
      (*env)->DeleteLocalRef(env, app);
      app = NULL;
    }

    /* Resolve application using multiple framework entry points. */
    app = resolve_application_jni(env, cls_at);

    if (app) {
      resolved_any_app = true;
      char candidate_path[PATH_MAX] = {0};

      /* Try to extract this application's files directory. */
      if (extract_files_dir_from_app(env, app, candidate_path,
                                      sizeof(candidate_path))) {
        /* Check if this application has the payload marker. */
        if (is_target_app_ready(candidate_path)) {
          /* SUCCESS: This is the target application with payload. */
          strncpy(out_buf, candidate_path, out_sz - 1);
          out_buf[out_sz - 1] = '\0';
          result = 1;
          LOGI("get_files_dir_jni: target application found at '%s'", out_buf);
          break;
        }

        /* Candidate lacks payload; keep polling. Do NOT accept. */
        LOGD("get_files_dir_jni: candidate '%s' lacks payload, waiting for target",
             candidate_path);
      }

      app = NULL; /* Clear local ref, will re-fetch next iteration. */
    } else {
      LOGD("get_files_dir_jni: application not resolved yet, retrying...");
    }

    /* Log progress at intervals. */
    if (i % 5 == 0)
      LOGD("get_files_dir_jni: polling for target application with payload (%d/30)...", i);

    sleep(1);
  }

  /* Timeout reached without finding target. */
  if (!result) {
    if (resolved_any_app)
      LOGW("get_files_dir_jni: application resolved, but no payload marker found within timeout");
    else
      LOGW("get_files_dir_jni: no application resolved within timeout");
    goto jni_done;
  }

jni_done:
  /* Clean up all JNI references. */
  if (app)
    (*env)->DeleteLocalRef(env, app);
  if (cls_at)
    (*env)->DeleteLocalRef(env, cls_at);
  if (attached)
    (*jvm)->DetachCurrentThread(jvm);

  return result;
  #undef JNI_CLR
}