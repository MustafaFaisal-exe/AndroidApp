package com.example.cameraapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import androidx.core.content.ContextCompat

/**
 * Reference-style renderer: draw YOLO boxes directly onto a bitmap instead of relying
 * on view-space coordinate transforms.
 */
class YoloBoxRenderer(private val context: Context) {

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 32f
        typeface = Typeface.MONOSPACE
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeWidth = 2f
    }
    private val textBounds = Rect()

    fun renderTransparent(width: Int, height: Int, boxes: List<DetectionOverlay.Box>): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        renderOn(bitmap, boxes)
        return bitmap
    }

    fun renderOn(bitmap: Bitmap, boxes: List<DetectionOverlay.Box>) {
        val canvas = Canvas(bitmap)

        for (box in boxes) {
            val left = box.x1.coerceIn(0f, bitmap.width.toFloat())
            val top = box.y1.coerceIn(0f, bitmap.height.toFloat())
            val right = box.x2.coerceIn(0f, bitmap.width.toFloat())
            val bottom = box.y2.coerceIn(0f, bitmap.height.toFloat())
            if (right <= left || bottom <= top) continue

            boxPaint.color = box.color
            boxPaint.strokeWidth = box.strokeWidth

            canvas.drawRect(left, top, right, bottom, outlinePaint)
            canvas.drawRect(left, top, right, bottom, boxPaint)

            if (box.label.isNotEmpty()) {
                labelPaint.getTextBounds(box.label, 0, box.label.length, textBounds)
                val pad = 6f
                val textW = labelPaint.measureText(box.label)
                val textH = textBounds.height().coerceAtLeast(labelPaint.textSize.toInt()).toFloat()
                labelBgPaint.color = box.color
                canvas.drawRect(
                    left,
                    (top - textH - pad * 2).coerceAtLeast(0f),
                    left + textW + pad * 2,
                    top,
                    labelBgPaint
                )
                canvas.drawText(box.label, left + pad, (top - pad).coerceAtLeast(labelPaint.textSize), labelPaint)
            }
        }
    }
}
