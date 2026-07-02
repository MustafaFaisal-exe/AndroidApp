package com.example.cameraapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.WindowManager
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
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
    private lateinit var denoiseRow: android.view.View
    private lateinit var brightnessRow: android.view.View
    private lateinit var contrastRow: android.view.View
    private lateinit var brightnessSlider: Slider
    private lateinit var contrastSlider: Slider
    private lateinit var brightnessValueText: TextView
    private lateinit var contrastValueText: TextView
    private lateinit var truckConfidenceText: TextView
    private lateinit var loadStatusBadge: TextView
    private lateinit var heatmapView: ImageView
    private lateinit var debugMetricsText: TextView
    private lateinit var stabilityProgressText: TextView
    private lateinit var analyzingOverlay: android.view.View
    private lateinit var retryButton: MaterialButton

    // ─── State ───────────────────────────────────────────────────────
    enum class CaptureState { SCANNING, STABILIZING, ANALYZING, RESULT }
    private var currentState = CaptureState.SCANNING

    private var currentBase    = "bc"
    private var currentOverlay = "yolo"
    private val processing     = AtomicBoolean(false)
    private var totalFrameCount = 0
    private var lastAnalysis: TruckAnalysisResult? = null
    private var lastBoxes: List<DetectionOverlay.Box> = emptyList()
    private var lastSourceW = 0
    private var lastSourceH = 0

    // Configurable parameters
    private var marginLeft = 80
    private var marginRight = 80
    private var marginTop = 80
    private var marginBottom = 80

    // ─── Camera / processing ─────────────────────────────────────────
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var frameProcessor: FrameProcessor
    private lateinit var depthEstimator: DepthEstimator
    private lateinit var truckClassifier: TruckClassifier
    private lateinit var analyzer: TruckLoadAnalyzer
    private lateinit var stabilityTracker: StabilityTracker
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var camera: Camera? = null

    private val backCallback = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() {
            resetScanning()
        }
    }

    // Persistent buffers to avoid GC pressure
    private var rgbaMatCached: Mat? = null
    private var rawFrameCached: Mat? = null
    private var rotatedFrameCached: Mat? = null

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
        
        cameraExecutor = Executors.newSingleThreadExecutor()
        frameProcessor = FrameProcessor(this)
        depthEstimator = DepthEstimator(this)
        truckClassifier = TruckClassifier(this)
        analyzer = TruckLoadAnalyzer(frameProcessor.yolo, depthEstimator, truckClassifier)
        stabilityTracker = StabilityTracker()
        
        onBackPressedDispatcher.addCallback(this, backCallback)
        retryButton.setOnClickListener { resetScanning() }
        
        setupModeToggles()

        requestCameraPermission()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        frameProcessor.close()
        depthEstimator.close()
        truckClassifier.close()
        rgbaMatCached?.release()
        rawFrameCached?.release()
        rotatedFrameCached?.release()
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
        denoiseRow       = findViewById(R.id.denoiseRow)
        brightnessRow    = findViewById(R.id.brightnessRow)
        contrastRow      = findViewById(R.id.contrastRow)
        brightnessSlider = findViewById(R.id.brightnessSlider)
        contrastSlider   = findViewById(R.id.contrastSlider)
        brightnessValueText = findViewById(R.id.brightnessValue)
        contrastValueText   = findViewById(R.id.contrastValue)
        truckConfidenceText = findViewById(R.id.truckConfidenceText)
        loadStatusBadge     = findViewById(R.id.loadStatusBadge)
        heatmapView         = findViewById(R.id.heatmapView)
        debugMetricsText    = findViewById(R.id.debugMetricsText)
        stabilityProgressText = findViewById(R.id.stabilityProgressText)
        analyzingOverlay    = findViewById(R.id.analyzingOverlay)
        retryButton         = findViewById(R.id.retryButton)
    }

    // ─────────────────────────────────────────────────────────────────
    // Mode toggles
    // ─────────────────────────────────────────────────────────────────

    private fun setupModeToggles() {
        baseModeGroup.check(R.id.btnBC)

        baseModeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            currentBase = when (checkedId) {
                R.id.btnPipeline -> "pipeline"
                else             -> "bc"
            }
            onModeChanged()
        }

        setupSliders()
        onModeChanged() // Initialize state
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

        if (!frameProcessor.yolo.isReady()) {
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
        }
        
        if (!depthEstimator.isReady() && depthEstimator.status != DepthEstimator.Status.LOADING) {
            if (depthEstimator.status == DepthEstimator.Status.ERROR) {
                setStatus("Depth Model Error: ${depthEstimator.errorMessage}", StatusLevel.ERROR)
            } else {
                setStatus("Loading Depth Model…", StatusLevel.WARN)
                depthEstimator.loadAsync {
                    mainHandler.post {
                        if (depthEstimator.isReady()) {
                            setStatus("Depth Model Ready ✓", StatusLevel.OK)
                        } else {
                            setStatus("Depth Init Failed", StatusLevel.ERROR)
                        }
                    }
                }
            }
        }

        if (!truckClassifier.isReady() && truckClassifier.status != TruckClassifier.Status.LOADING) {
            if (truckClassifier.status == TruckClassifier.Status.ERROR) {
                setStatus("Classifier Error: ${truckClassifier.errorMessage}", StatusLevel.ERROR)
            } else {
                truckClassifier.loadAsync {
                    mainHandler.post {
                        if (truckClassifier.isReady()) {
                            setStatus("Classifier Ready ✓", StatusLevel.OK)
                        } else {
                            setStatus("Classifier Init Failed", StatusLevel.ERROR)
                        }
                    }
                }
            }
        }

        previewView.visibility = android.view.View.VISIBLE
        processedView.visibility = android.view.View.VISIBLE
        processedView.scaleType = ImageView.ScaleType.FIT_CENTER
        processedView.setBackgroundColor(android.graphics.Color.BLACK)
        detectionOverlay.visibility = android.view.View.VISIBLE
        
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

        listOf(R.id.btnBC, R.id.btnPipeline).forEach { id ->
            val btn = findViewById<MaterialButton>(id)
            val sel = baseModeGroup.checkedButtonId == id
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

        if (currentState == CaptureState.RESULT || currentState == CaptureState.ANALYZING) {
            proxy.close()
            processing.set(false)
            return
        }

        try {
            val rotationDegrees = proxy.imageInfo.rotationDegrees
            val rgbaMat = imageProxyToRgbaMat(proxy)
            
            val rawFrame = rawFrameCached?.takeIf { it.rows() == rgbaMat.rows() && it.cols() == rgbaMat.cols() }
                ?: Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3).also { rawFrameCached = it }
                
            Imgproc.cvtColor(rgbaMat, rawFrame, Imgproc.COLOR_RGBA2BGR)

            totalFrameCount++

            val frame = if (rotationDegrees != 0) {
                val rotated = rotatedFrameCached?.let { 
                    if (rotationDegrees % 180 == 0) {
                        if (it.rows() == rawFrame.rows() && it.cols() == rawFrame.cols()) it else null
                    } else {
                        if (it.rows() == rawFrame.cols() && it.cols() == rawFrame.rows()) it else null
                    }
                } ?: Mat().also { rotatedFrameCached = it }
                
                when (rotationDegrees) {
                    90  -> Core.rotate(rawFrame, rotated, Core.ROTATE_90_CLOCKWISE)
                    180 -> Core.rotate(rawFrame, rotated, Core.ROTATE_180)
                    270 -> Core.rotate(rawFrame, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)
                    else -> rawFrame.copyTo(rotated)
                }
                rotated
            } else rawFrame

            val resultMat = frameProcessor.process(frame, currentBase, currentOverlay)
            val sourceW = resultMat.cols()
            val sourceH = resultMat.rows()

            // YOLO continues running every frame
            frameProcessor.yolo.infer(resultMat)
            val target = frameProcessor.yolo.lastTargetDetection
            
            if (target != null) {
                stabilityTracker.addDetection(
                    StabilityTracker.DetectionEntry(
                        target.conf,
                        android.graphics.RectF(target.x1, target.y1, target.x2, target.y2),
                        target.classId
                    )
                )
                if (currentState == CaptureState.SCANNING) {
                    updateUiForState(CaptureState.STABILIZING)
                }
            } else {
                stabilityTracker.addDetection(null)
                if (currentState == CaptureState.STABILIZING && stabilityTracker.getProgress() == 0) {
                    updateUiForState(CaptureState.SCANNING)
                }
            }

            if (currentState == CaptureState.STABILIZING) {
                val progress = stabilityTracker.getProgress()
                val max = stabilityTracker.getMaxProgress()
                mainHandler.post {
                    stabilityProgressText.text = "Hold steady... $progress/$max"
                }
                
                if (stabilityTracker.isStable()) {
                    triggerCapture(resultMat)
                }
            }

            val boxes = frameProcessor.yolo.lastBoxes
            val outBitmap = ImageProcessor.matToBitmap(resultMat)
            
            // Only release if it's a new Mat from processor, not our persistent buffers
            if (resultMat !== frame && resultMat !== rawFrame && resultMat !== rotatedFrameCached) {
                resultMat.release()
            }

            mainHandler.post {
                detectionOverlay.setDetections(
                    boxes, sourceW, sourceH,
                    BoxCoordinateMapper.ScaleType.FIT_CENTER
                )
                if (currentState != CaptureState.RESULT && currentState != CaptureState.ANALYZING) {
                    processedView.setImageBitmap(outBitmap)
                }
                truckConfidenceText.text = buildTruckConfidenceLabel()
                updateFps()
            }
        } catch (e: Exception) {
            mainHandler.post { setStatus("Error: ${e.message}", StatusLevel.ERROR) }
        } finally {
            proxy.close()
            processing.set(false)
        }
    }

    private fun triggerCapture(frame: Mat) {
        if (currentState == CaptureState.ANALYZING || currentState == CaptureState.RESULT) return
        
        lastSourceW = frame.cols()
        lastSourceH = frame.rows()
        
        updateUiForState(CaptureState.ANALYZING)
        
        // Freeze-frame
        val frozenBitmap = ImageProcessor.matToBitmap(frame)
        val frameCopy = frame.clone()

        mainHandler.post {
            processedView.setImageBitmap(frozenBitmap)
            processedView.visibility = android.view.View.VISIBLE
        }

        Thread {
            try {
                val analysis = analyzer.analyze(
                    frameCopy,
                    marginLeft, marginRight, marginTop, marginBottom
                )
                lastAnalysis = analysis

                lastBoxes = if (analysis.source == "yolo") {
                    frameProcessor.yolo.lastBoxes
                } else {
                    listOf(
                        DetectionOverlay.Box(
                            analysis.cropRect.left.toFloat(),
                            analysis.cropRect.top.toFloat(),
                            analysis.cropRect.right.toFloat(),
                            analysis.cropRect.bottom.toFloat(),
                            "FALLBACK",
                            android.graphics.Color.YELLOW,
                            4f
                        )
                    )
                }
                
                mainHandler.post {
                    displayAnalysisResult(analysis)
                    updateUiForState(CaptureState.RESULT)
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Analysis failed", e)
                mainHandler.post {
                    setStatus("Analysis failed: ${e.message}", StatusLevel.ERROR)
                    resetScanning()
                }
            } finally {
                frameCopy.release()
            }
        }.start()
    }

    private fun resetScanning() {
        stabilityTracker.reset()
        lastAnalysis = null
        lastBoxes = emptyList()
        updateUiForState(CaptureState.SCANNING)
    }

    private fun updateUiForState(state: CaptureState) {
        currentState = state
        mainHandler.post {
            when (state) {
                CaptureState.SCANNING -> {
                    stabilityProgressText.visibility = android.view.View.GONE
                    analyzingOverlay.visibility = android.view.View.GONE
                    retryButton.visibility = android.view.View.GONE
                    loadStatusBadge.text = "SCANNING"
                    loadStatusBadge.backgroundTintList = android.content.res.ColorStateList.valueOf(android.graphics.Color.DKGRAY)
                    heatmapView.setImageBitmap(null)
                    processedView.visibility = android.view.View.VISIBLE
                    detectionOverlay.visibility = android.view.View.VISIBLE
                    backCallback.isEnabled = false
                }
                CaptureState.STABILIZING -> {
                    stabilityProgressText.visibility = android.view.View.VISIBLE
                    analyzingOverlay.visibility = android.view.View.GONE
                    retryButton.visibility = android.view.View.GONE
                    backCallback.isEnabled = false
                }
                CaptureState.ANALYZING -> {
                    stabilityProgressText.visibility = android.view.View.GONE
                    analyzingOverlay.visibility = android.view.View.VISIBLE
                    retryButton.visibility = android.view.View.GONE
                    backCallback.isEnabled = false
                }
                CaptureState.RESULT -> {
                    stabilityProgressText.visibility = android.view.View.GONE
                    analyzingOverlay.visibility = android.view.View.GONE
                    retryButton.visibility = android.view.View.VISIBLE
                    processedView.visibility = android.view.View.VISIBLE
                    detectionOverlay.visibility = android.view.View.VISIBLE
                    backCallback.isEnabled = true
                }
            }
        }
    }

    private fun displayAnalysisResult(analysis: TruckAnalysisResult) {
        loadStatusBadge.text = analysis.status
        loadStatusBadge.backgroundTintList = android.content.res.ColorStateList.valueOf(
            when {
                analysis.status.contains("ERROR") -> android.graphics.Color.DKGRAY
                analysis.isFull -> android.graphics.Color.GREEN
                else -> android.graphics.Color.RED
            }
        )
        heatmapView.setImageBitmap(analysis.depthMapBitmap)
        debugMetricsText.text = "Mean: %.1f Std: %.1f Src: %s".format(
            analysis.depthMean, analysis.depthStd, analysis.source
        )

        detectionOverlay.setDetections(
            lastBoxes, lastSourceW, lastSourceH,
            BoxCoordinateMapper.ScaleType.FIT_CENTER
        )

        if (depthEstimator.status != DepthEstimator.Status.ERROR && 
            frameProcessor.yolo.status != YoloDetector.Status.ERROR) {
            when {
                analysis.source == "yolo" -> setStatus("Detected via YOLO", StatusLevel.OK)
                else -> setStatus("Detected via center-crop fallback", StatusLevel.WARN)
            }
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
        val width       = proxy.width
        val height      = proxy.height
        val rgba        = rgbaMatCached?.takeIf { it.rows() == height && it.cols() == width }
            ?: Mat(height, width, CvType.CV_8UC4).also { rgbaMatCached = it }
            
        val plane       = proxy.planes[0]
        val buffer      = plane.buffer
        buffer.rewind()
        val rowStride   = plane.rowStride
        val pixelStride = plane.pixelStride

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