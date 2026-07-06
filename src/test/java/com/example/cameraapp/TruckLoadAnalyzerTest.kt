package com.example.cameraapp

import android.graphics.Rect
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class TruckLoadAnalyzerTest {

    @Test
    fun testFallbackCropCalculation() {
        val w = 1000
        val h = 1000
        val marginLeft = 100
        val marginRight = 200
        val marginTop = 50
        val marginBottom = 150
        
        val rect = TruckLoadAnalyzer.calculateFallbackCrop(w, h, marginLeft, marginRight, marginTop, marginBottom)
        
        assertEquals(100, rect.left)
        assertEquals(50, rect.top)
        assertEquals(800, rect.right)
        assertEquals(850, rect.bottom)
    }

    @Test
    fun testEdgeCaseMargins() {
        val w = 100
        val h = 100
        // Margins too large
        val rect = TruckLoadAnalyzer.calculateFallbackCrop(w, h, 60, 60, 0, 0)
        
        assertEquals(60, rect.left)
        assertEquals(61, rect.right)
        assertEquals(1, rect.width())
    }
}
