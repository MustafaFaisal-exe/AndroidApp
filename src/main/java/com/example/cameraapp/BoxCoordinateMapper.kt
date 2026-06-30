package com.example.cameraapp

import android.graphics.RectF

/** Maps detection boxes from image pixel space to view space (accounts for letterboxing). */
object BoxCoordinateMapper {

    enum class ScaleType { FIT_CENTER, FILL_CENTER }

    data class Transform(
        val scale: Float,
        val offsetX: Float,
        val offsetY: Float,
        val imageWidth: Int,
        val imageHeight: Int
    )

    fun computeTransform(
        imageWidth: Int,
        imageHeight: Int,
        viewWidth: Float,
        viewHeight: Float,
        scaleType: ScaleType
    ): Transform {
        if (imageWidth <= 0 || imageHeight <= 0 || viewWidth <= 0 || viewHeight <= 0) {
            return Transform(1f, 0f, 0f, imageWidth, imageHeight)
        }
        val sx = viewWidth / imageWidth
        val sy = viewHeight / imageHeight
        val scale = when (scaleType) {
            ScaleType.FIT_CENTER  -> minOf(sx, sy)
            ScaleType.FILL_CENTER -> maxOf(sx, sy)
        }
        val dispW = imageWidth * scale
        val dispH = imageHeight * scale
        return Transform(
            scale = scale,
            offsetX = (viewWidth - dispW) / 2f,
            offsetY = (viewHeight - dispH) / 2f,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    }

    fun mapRect(box: RectF, transform: Transform): RectF {
        return RectF(
            box.left * transform.scale + transform.offsetX,
            box.top * transform.scale + transform.offsetY,
            box.right * transform.scale + transform.offsetX,
            box.bottom * transform.scale + transform.offsetY
        )
    }
}
