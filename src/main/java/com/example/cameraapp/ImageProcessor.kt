package com.example.cameraapp

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

object ImageProcessor {

    // Default sharpening kernel — identity kernel (no change)
    val SHARPEN_KERNEL: Mat by lazy {
        Mat(3, 3, CvType.CV_32F).apply {
            put(0, 0,
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0)
        }
    }

    fun preprocessBC(frame: Mat): Mat {
        val result = Mat()
        frame.copyTo(result)
        return result
    }

    // Reusable CLAHE object to avoid allocations
    private val claheObj by lazy { Imgproc.createCLAHE(1.5, Size(4.0, 4.0)) }

    fun preprocess(
        frame: Mat,
        gamma: Double = 1.8,
        clipLimit: Double = 1.5,
        tileSize: Size = Size(4.0, 4.0),
        sharpenKernel: Mat = SHARPEN_KERNEL
    ): Mat {
        // 1. Denoising - Box Blur Only
        var out = Mat()
        Imgproc.boxFilter(frame, out, -1, Size(5.0, 5.0))

        // 2. Gamma correction via LUT
        val lut = buildGammaLUT(gamma)
        val gammaOut = Mat()
        Core.LUT(out, lut, gammaOut)
        out.release()
        lut.release()
        out = gammaOut

        // 3. CLAHE on the L channel of LAB colour space
        val lab = Mat()
        Imgproc.cvtColor(out, lab, Imgproc.COLOR_BGR2Lab)
        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)

        // Using pre-allocated CLAHE object
        claheObj.apply(channels[0], channels[0])
        Core.merge(channels, lab)

        val claheOut = Mat()
        Imgproc.cvtColor(lab, claheOut, Imgproc.COLOR_Lab2BGR)
        out.release()
        lab.release()
        channels.forEach { it.release() }
        out = claheOut

        // 4. Sharpening
        val sharpened = Mat()
        Imgproc.filter2D(out, sharpened, -1, sharpenKernel)
        out.release()

        return sharpened
    }

    fun jpegToMat(bytes: ByteArray): Mat {
        val mat = Mat()
        val encoded = MatOfByte(*bytes)
        val decoded = org.opencv.imgcodecs.Imgcodecs.imdecode(encoded, org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR)
        encoded.release()
        return decoded
    }

    fun matToBitmap(mat: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_BGR2RGBA)
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        return bmp
    }

    fun bitmapToMat(bitmap: Bitmap): Mat {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        // bitmapToMat gives RGBA → convert to BGR
        val bgr = Mat()
        Imgproc.cvtColor(mat, bgr, Imgproc.COLOR_RGBA2BGR)
        mat.release()
        return bgr
    }

    private var cachedGamma: Double = -1.0
    private var cachedLut: Mat? = null

    private fun buildGammaLUT(gamma: Double): Mat {
        if (gamma == cachedGamma && cachedLut != null) {
            return cachedLut!!.clone()
        }
        
        val lut = Mat(1, 256, CvType.CV_8UC1)
        val invGamma = 1.0 / gamma
        val bytes = ByteArray(256)
        for (i in 0..255) {
            bytes[i] = (Math.pow(i / 255.0, invGamma) * 255.0 + 0.5).toInt().coerceIn(0, 255).toByte()
        }
        lut.put(0, 0, bytes)
        
        cachedGamma = gamma
        cachedLut?.release()
        cachedLut = lut.clone()
        return lut
    }
}
