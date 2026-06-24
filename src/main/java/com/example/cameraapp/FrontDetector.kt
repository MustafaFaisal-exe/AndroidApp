package com.example.cameraapp

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

object FrontDetector {

    private const val TAG = "FrontDetector"

    fun IsFront(crop: Mat): TruckView = classify(crop)

    fun classify(crop: Mat): TruckView {
        val h = crop.rows()
        val w = crop.cols()
        if (h <= 0 || w <= 0) return TruckView.REAR

        // ── Side view early exit ──────────────────────────────────────────────────
        if (w.toDouble() / h.toDouble() > 1.6) return TruckView.SIDE

        val gray = Mat()
        val hsv  = Mat()
        Imgproc.cvtColor(crop, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.cvtColor(crop, hsv,  Imgproc.COLOR_BGR2HSV)

        val cropMean = Core.mean(gray).`val`[0]

        return try {
            val hArr = ByteArray(h * w)
            val sArr = ByteArray(h * w)
            val vArr = ByteArray(h * w)
            val channels = mutableListOf<Mat>()
            Core.split(hsv, channels)
            channels[0].get(0, 0, hArr)
            channels[1].get(0, 0, sArr)
            channels[2].get(0, 0, vArr)
            channels.forEach { it.release() }

            // Wider hue/sat range: (hh < 10 || hh > 170) -> (hh < 10 is 0-20 deg, hh > 170 is 340-360 deg)
            // Python: ((h_ch < 20) | (h_ch > 160)) & (s_ch > 60) & (v_ch > 50)
            // OpenCV H: 0-179, S: 0-255, V: 0-255
            val redMask = BooleanArray(h * w) { i ->
                val hh = (hArr[i].toInt() and 0xFF).toFloat()  // 0-179
                val ss = (sArr[i].toInt() and 0xFF).toFloat()  // 0-255
                val vv = (vArr[i].toInt() and 0xFF).toFloat()  // 0-255
                (hh < 10f || hh > 170f) && ss > 60f && vv > 50f
            }

            var score = 0

            // ── Red veto: lopsided dominant red = tail light, balanced red = decoration ──
            val redL = redMaskMean(redMask, w, h, 0.45, 0.72, 0.00, 0.15)
            val redR = redMaskMean(redMask, w, h, 0.45, 0.72, 0.85, 1.00)
            val redMax = maxOf(redL, redR)
            val redDiff = Math.abs(redL - redR)

            if      (redMax > 0.50 && redDiff > 0.20) score -= 5
            else if (redMax > 0.35 && redDiff > 0.20) score -= 4
            else if (redMax > 0.25 && redDiff > 0.15) score -= 3
            else if (redMax > 0.14 && redDiff > 0.06) score -= 3  // ← was -1, now -3; catches symmetric tail lights
            else if (redMax > 0.14)                   score -= 2  // ← was -1

            // ── Signal 1: Top strip relative brightness ───────────────────────────────
            val topRel = regionMean(gray, 0.00, 0.12, 0.10, 0.90) - cropMean
            if      (topRel > 30)  score += 2
            else if (topRel > 10)  score += 1
            else                   score -= 1

            // ── Signal 2: Upper center relative darkness ──────────────────────────────
            val ucRel = cropMean - regionMean(gray, 0.10, 0.40, 0.20, 0.80)
            if      (ucRel > 25)   score += 3
            else if (ucRel > 8)    score += 2
            else if (ucRel > 0)    score += 1
            else if (ucRel > -10)  score -= 1
            else                   score -= 2

            // ── Signal 3: Headlight asymmetry ────────────────────────────────────────
            val hlL = regionMean(gray, 0.42, 0.68, 0.08, 0.34)
            val hlR = regionMean(gray, 0.42, 0.68, 0.66, 0.92)
            val hlDiff = Math.abs(hlL - hlR)
            if      (hlDiff > 30)  score += 3
            else if (hlDiff > 20)  score += 2
            else if (hlDiff > 12)  score += 1

            // ── Signal 4: Lower center relative darkness ──────────────────────────────
            val bumperRel = cropMean - regionMean(gray, 0.85, 1.00, 0.15, 0.85)
            if      (bumperRel > 60)   score += 4
            else if (bumperRel > 35)   score += 3
            else if (bumperRel > 15)   score += 1
            else if (bumperRel < -15)  score -= 2

            // ── Signal 5: Grille center band relative darkness ────────────────────────
            val grilleRel = cropMean - regionMean(gray, 0.55, 0.75, 0.30, 0.70)
            if      (grilleRel > 20)   score += 2
            else if (grilleRel > 8)    score += 1

            val result = if (score >= 3) TruckView.FRONT else TruckView.REAR

            Log.d(TAG, "[heuristic] mean=${"%.1f".format(cropMean)} score=$score " +
                    "topRel=${"%.1f".format(topRel)} ucRel=${"%.1f".format(ucRel)} " +
                    "bumperRel=${"%.1f".format(bumperRel)} grilleRel=${"%.1f".format(grilleRel)} " +
                    "hlDiff=${"%.1f".format(hlDiff)} redL=${"%.3f".format(redL)} " +
                    "redR=${"%.3f".format(redR)} redDiff=${"%.3f".format(redDiff)} → $result")

            return result

        } catch (e: Exception) {
            Log.e(TAG, "classify error", e)
            TruckView.REAR
        } finally {
            gray.release()
            hsv.release()
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Mean of a grayscale region using fractional coordinates (yStart, yEnd, xStart, xEnd). */
    private fun regionMean(
        mat: Mat,
        yStart: Double, yEnd: Double,
        xStart: Double, xEnd: Double
    ): Double {
        val h  = mat.rows(); val w = mat.cols()
        val rx1 = (xStart * w).toInt().coerceIn(0, w)
        val ry1 = (yStart * h).toInt().coerceIn(0, h)
        val rx2 = (xEnd   * w).toInt().coerceIn(rx1 + 1, w)
        val ry2 = (yEnd   * h).toInt().coerceIn(ry1 + 1, h)
        val region = mat.submat(ry1, ry2, rx1, rx2)
        val value  = Core.mean(region).`val`[0]
        region.release()
        return value
    }

    /** Mean of a boolean red-mask region using fractional coordinates. */
    private fun redMaskMean(
        mask: BooleanArray, w: Int, h: Int,
        yStart: Double, yEnd: Double,
        xStart: Double, xEnd: Double
    ): Double {
        val rx1 = (xStart * w).toInt().coerceIn(0, w)
        val rx2 = (xEnd   * w).toInt().coerceIn(rx1 + 1, w)
        val ry1 = (yStart * h).toInt().coerceIn(0, h)
        val ry2 = (yEnd   * h).toInt().coerceIn(ry1 + 1, h)
        var count = 0; var total = 0
        for (y in ry1 until ry2) for (x in rx1 until rx2) {
            if (mask[y * w + x]) count++
            total++
        }
        return if (total == 0) 0.0 else count.toDouble() / total
    }
}