package com.example.cameraapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.slider.Slider
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    // ─── Views ───────────────────────────────────────────────────────
    private lateinit var previewView: PreviewView
    private lateinit var processedView: ImageView
    private lateinit var detectionOverlay: DetectionOverlay
    private lateinit var modeBadge: TextView
    private lateinit var fpsText: TextView
    private lateinit var statusText: TextView
    private lateinit var statusDot: android.view.View
    private lateinit var baseModeGroup: MaterialButtonToggleGroup
    private lateinit var overlayModeGroup: MaterialButtonToggleGroup
    private lateinit var denoiseRow: android.view.View
    private lateinit var brightnessRow: android.view.View
    private lateinit var contrastRow: android.view.View
    private lateinit var brightnessSlider: Slider
    private lateinit var contrastSlider: Slider
    private lateinit var brightnessValueText: TextView
    private lateinit var contrastValueText: TextView
    private lateinit var truckConfidenceText: TextView

    // ─── State ───────────────────────────────────────────────────────
    private var currentBase    = "raw"
    private var currentOverlay = "none"
    private val processing     = AtomicBoolean(false)

    // ─── Camera / processing ─────────────────────────────────────────
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var frameProcessor: FrameProcessor
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var camera: Camera? = null

    // ─── FPS tracking ────────────────────────────────────────────────
    private val mainHandler = Handler(Looper.getMainLooper())

    // ─────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.systemUiVisibility = (
                android.view.View.SYSTEM_UI_FLAG_FULLSCREEN or
                        android.view.View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                        android.view.View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                )

        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show()
        }

        bindViews()
        setupModeToggles()

        cameraExecutor = Executors.newSingleThreadExecutor()
        frameProcessor = FrameProcessor(this)

        requestCameraPermission()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        frameProcessor.close()
    }

    // ─────────────────────────────────────────────────────────────────
    // View binding
    // ─────────────────────────────────────────────────────────────────

    private fun bindViews() {
        previewView      = findViewById(R.id.previewView)
        processedView    = findViewById(R.id.processedView)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        modeBadge        = findViewById(R.id.modeBadge)
        fpsText          = findViewById(R.id.fpsText)
        statusText       = findViewById(R.id.statusText)
        statusDot        = findViewById(R.id.statusDot)
        baseModeGroup    = findViewById(R.id.baseModeGroup)
        overlayModeGroup = findViewById(R.id.overlayModeGroup)
        denoiseRow       = findViewById(R.id.denoiseRow)
        brightnessRow    = findViewById(R.id.brightnessRow)
        contrastRow      = findViewById(R.id.contrastRow)
        brightnessSlider = findViewById(R.id.brightnessSlider)
        contrastSlider   = findViewById(R.id.contrastSlider)
        brightnessValueText = findViewById(R.id.brightnessValue)
        contrastValueText   = findViewById(R.id.contrastValue)
        truckConfidenceText = findViewById(R.id.truckConfidenceText)
    }

    // ─────────────────────────────────────────────────────────────────
    // Mode toggles
    // ─────────────────────────────────────────────────────────────────

    private fun setupModeToggles() {
        baseModeGroup.check(R.id.btnRaw)
        overlayModeGroup.check(R.id.btnNone)

        baseModeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            currentBase = when (checkedId) {
                R.id.btnBC       -> "bc"
                R.id.btnPipeline -> "pipeline"
                else             -> "raw"
            }
            onModeChanged()
        }

        overlayModeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            currentOverlay = when (checkedId) {
                R.id.btnYolo -> "yolo"
                else         -> "none"
            }
            onModeChanged()
        }

        setupSliders()
    }

    private fun setupSliders() {
        configureSlider(brightnessSlider, -10f, 10f, 1f, 0f) { value ->
            frameProcessor.brightness = value.toInt()
            brightnessValueText.text = value.toInt().toString()
        }
        configureSlider(contrastSlider, 0.0f, 5.0f, 0.1f, 1.0f) { value ->
            frameProcessor.contrast = value.toDouble()
            contrastValueText.text = "%.1f".format(value)
        }
    }

    private fun updateExposure() {
        val control = camera?.cameraControl ?: return
        if (currentBase == "bc") {
            control.setExposureCompensationIndex(-8)
        } else {
            control.setExposureCompensationIndex(0)
        }
    }

    private fun configureSlider(
        slider: Slider,
        min: Float,
        max: Float,
        step: Float,
        initial: Float,
        onChanged: (Float) -> Unit
    ) {
        slider.valueFrom = min
        slider.valueTo = max
        slider.stepSize = step
        slider.value = initial
        slider.addOnChangeListener { _, value, _ -> onChanged(value) }
    }

    private fun onModeChanged() {
        val mode = "$currentBase+$currentOverlay"
        modeBadge.text = mode

        if (currentOverlay == "yolo" && !frameProcessor.yolo.isReady()) {
            setStatus("Loading YOLO…", StatusLevel.WARN)
            frameProcessor.yolo.loadAsync {
                mainHandler.post {
                    if (frameProcessor.yolo.isReady()) {
                        setStatus("YOLO ready ✓", StatusLevel.OK)
                    } else {
                        setStatus("YOLO Error: ${frameProcessor.yolo.errorMessage}", StatusLevel.ERROR)
                    }
                }
            }
        } else if (currentOverlay != "yolo") {
            setStatus("Mode: $mode", StatusLevel.OK)
        }

        val showProcessed = currentBase != "raw"
        val showYolo = currentOverlay == "yolo"

        previewView.visibility = android.view.View.VISIBLE
        processedView.visibility = if (showProcessed) android.view.View.VISIBLE else android.view.View.GONE
        processedView.scaleType = ImageView.ScaleType.FIT_CENTER
        processedView.setBackgroundColor(android.graphics.Color.BLACK)
        detectionOverlay.visibility = if (showYolo) android.view.View.VISIBLE else android.view.View.GONE
        detectionOverlay.clear()
        if (!showYolo) truckConfidenceText.text = ""

        denoiseRow.visibility = if (currentBase == "pipeline") android.view.View.VISIBLE
        else android.view.View.GONE

        val showSliders = (currentBase == "bc")
        brightnessRow.visibility = if (showSliders) android.view.View.VISIBLE else android.view.View.GONE
        contrastRow.visibility   = if (showSliders) android.view.View.VISIBLE else android.view.View.GONE

        updateExposure()
        highlightToggles()
    }

    private fun highlightToggles() {
        val activeColor   = ContextCompat.getColor(this, R.color.colorPrimary)
        val inactiveColor = ContextCompat.getColor(this, R.color.colorTextMuted)

        listOf(R.id.btnRaw, R.id.btnBC, R.id.btnPipeline).forEach { id ->
            val btn = findViewById<MaterialButton>(id)
            val sel = baseModeGroup.checkedButtonId == id
            btn.setTextColor(if (sel) activeColor else inactiveColor)
            btn.strokeColor = if (sel)
                ContextCompat.getColorStateList(this, R.color.colorPrimary)
            else
                ContextCompat.getColorStateList(this, R.color.colorDivider)
        }
        listOf(R.id.btnNone, R.id.btnYolo).forEach { id ->
            val btn = findViewById<MaterialButton>(id)
            val sel = overlayModeGroup.checkedButtonId == id
            btn.setTextColor(if (sel) activeColor else inactiveColor)
            btn.strokeColor = if (sel)
                ContextCompat.getColorStateList(this, R.color.colorPrimary)
            else
                ContextCompat.getColorStateList(this, R.color.colorDivider)
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Camera permission
    // ─────────────────────────────────────────────────────────────────

    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else Toast.makeText(this, getString(R.string.camera_permission_denied),
            Toast.LENGTH_LONG).show()
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else permLauncher.launch(Manifest.permission.CAMERA)
    }

    // ─────────────────────────────────────────────────────────────────
    // CameraX setup
    // ─────────────────────────────────────────────────────────────────

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindCamera()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(960, 540))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { ia -> ia.setAnalyzer(cameraExecutor, ::analyzeFrame) }

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
            )
            setStatus("Camera ready", StatusLevel.OK)
            updateExposure()
        } catch (e: Exception) {
            setStatus("Camera error: ${e.message}", StatusLevel.ERROR)
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Per-frame analysis
    // ─────────────────────────────────────────────────────────────────

    private fun analyzeFrame(proxy: ImageProxy) {
        if (!processing.compareAndSet(false, true)) {
            proxy.close(); return
        }

        try {
            if (currentBase == "raw" && currentOverlay == "none") {
                mainHandler.post { detectionOverlay.clear() }
                updateFps(); proxy.close(); processing.set(false); return
            }

            val rotationDegrees = proxy.imageInfo.rotationDegrees
            val rgbaMat = imageProxyToRgbaMat(proxy)
            val rawFrame = Mat()
            Imgproc.cvtColor(rgbaMat, rawFrame, Imgproc.COLOR_RGBA2BGR)
            rgbaMat.release()

            val frame = if (rotationDegrees != 0) {
                val rotated = Mat()
                when (rotationDegrees) {
                    90  -> Core.rotate(rawFrame, rotated, Core.ROTATE_90_CLOCKWISE)
                    180 -> Core.rotate(rawFrame, rotated, Core.ROTATE_180)
                    270 -> Core.rotate(rawFrame, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)
                    else -> rawFrame.copyTo(rotated).let { rotated }
                }
                rawFrame.release(); rotated
            } else rawFrame

            val result = frameProcessor.process(frame, currentBase, currentOverlay)

            if (currentOverlay == "yolo") {
                val sourceW = result.cols()
                val sourceH = result.rows()
                val boxes = frameProcessor.yolo.lastBoxes

                val outBitmap = ImageProcessor.matToBitmap(result)
                if (result !== frame) result.release()
                frame.release()

                mainHandler.post {
                    detectionOverlay.setDetections(
                        boxes, sourceW, sourceH,
                        BoxCoordinateMapper.ScaleType.FIT_CENTER
                    )
                    processedView.setImageBitmap(outBitmap)
                    truckConfidenceText.text = buildTruckConfidenceLabel()
                    updateFps()
                    when {
                        frameProcessor.yolo.lastTargetDetection != null -> setStatus("Truck/bus detected", StatusLevel.OK)
                        else -> setStatus("No truck/bus detected", StatusLevel.WARN)
                    }
                }
                return
            }

            val outBitmap = ImageProcessor.matToBitmap(result)
            if (result !== frame) result.release()
            frame.release()

            mainHandler.post {
                detectionOverlay.setDetections(emptyList(), 1, 1, BoxCoordinateMapper.ScaleType.FIT_CENTER)
                processedView.setImageBitmap(outBitmap)
                updateFps()
            }

        } catch (e: Exception) {
            mainHandler.post { setStatus("Error: ${e.message}", StatusLevel.ERROR) }
        } finally {
            proxy.close()
            processing.set(false)
        }
    }

    private fun buildTruckConfidenceLabel(): String {
        val conf = frameProcessor.yolo.lastTargetConfidence ?: return ""
        val classId = frameProcessor.yolo.lastTargetClassId
        val view = frameProcessor.yolo.lastTruckView
        val name = when (classId) {
            5 -> "BUS"; 7 -> "TRUCK"; else -> "VEHICLE"
        }
        val viewStr = when (view) {
            TruckView.FRONT -> " (FRONT)"
            TruckView.REAR -> " (REAR)"
            TruckView.SIDE -> " (SIDE)"
            else -> ""
        }
        return "$name ${"%.0f".format(conf * 100)}%$viewStr"
    }

    // ─────────────────────────────────────────────────────────────────
    // FPS counter
    // ─────────────────────────────────────────────────────────────────

    private var fpsFrameCount  = 0
    private var fpsWindowStart = System.currentTimeMillis()
    private var rgbaRowBuffer  = ByteArray(0)

    private fun updateFps() {
        fpsFrameCount++
        val now     = System.currentTimeMillis()
        val elapsed = now - fpsWindowStart
        if (elapsed >= 1000L) {
            val fps = fpsFrameCount * 1000f / elapsed
            mainHandler.post { fpsText.text = "${"%.1f".format(fps)} fps" }
            fpsFrameCount  = 0
            fpsWindowStart = now
        }
    }

    private fun imageProxyToRgbaMat(proxy: ImageProxy): Mat {
        val plane       = proxy.planes[0]
        val buffer      = plane.buffer
        buffer.rewind()
        val width       = proxy.width
        val height      = proxy.height
        val rowStride   = plane.rowStride
        val pixelStride = plane.pixelStride
        val rgba        = Mat(height, width, CvType.CV_8UC4)

        if (pixelStride == 4 && rowStride == width * 4) {
            val expected = width * height * 4
            if (rgbaRowBuffer.size < expected) rgbaRowBuffer = ByteArray(expected)
            buffer.get(rgbaRowBuffer, 0, expected)
            rgba.put(0, 0, rgbaRowBuffer, 0, expected)
            return rgba
        }

        val rowData = ByteArray(rowStride)
        val dst     = ByteArray(width * 4)
        for (row in 0 until height) {
            buffer.get(rowData, 0, rowStride)
            var srcIndex = 0; var dstIndex = 0
            for (col in 0 until width) {
                dst[dstIndex++] = rowData[srcIndex]
                dst[dstIndex++] = rowData[srcIndex + 1]
                dst[dstIndex++] = rowData[srcIndex + 2]
                dst[dstIndex++] = rowData[srcIndex + 3]
                srcIndex += pixelStride
            }
            rgba.put(row, 0, dst)
        }
        return rgba
    }

    // ─────────────────────────────────────────────────────────────────
    // Status bar
    // ─────────────────────────────────────────────────────────────────

    enum class StatusLevel { OK, WARN, ERROR }

    private fun setStatus(msg: String, level: StatusLevel) {
        mainHandler.post {
            statusText.text = msg
            val dotColor = when (level) {
                StatusLevel.OK    -> ContextCompat.getColor(this, R.color.colorStatusOk)
                StatusLevel.WARN  -> ContextCompat.getColor(this, R.color.colorStatusWarn)
                StatusLevel.ERROR -> ContextCompat.getColor(this, R.color.colorStatusError)
            }
            (statusDot.background as? android.graphics.drawable.GradientDrawable)
                ?.setColor(dotColor)
                ?: statusDot.setBackgroundColor(dotColor)
        }
    }
}