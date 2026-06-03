package com.aimvision

import android.content.Context
import android.graphics.*
import android.view.View
import kotlin.math.min

class OverlayView(context: Context) : View(context) {

    private var detections: List<EnemyBox> = emptyList()
    private var srcW: Int = 1
    private var srcH: Int = 1

    // --- Paints ---
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 3.5f
    }

    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.FILL
        alpha = 18
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.FILL
        alpha = 220
    }

    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f
        typeface = Typeface.DEFAULT_BOLD
        letterSpacing = 0.08f
    }

    private val crosshairPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 2f
        alpha = 160
    }

    private val aimCirclePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF4488FF")
        style = Paint.Style.STROKE
        strokeWidth = 2f
        alpha = 180
    }

    private val scanPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1100FF44")
        style = Paint.Style.FILL
    }

    private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CC000000")
        style = Paint.Style.FILL
    }

    private val hudTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF00FF44")
        textSize = 26f
        typeface = Typeface.DEFAULT_BOLD
        letterSpacing = 0.12f
    }

    private var scanY = 0f
    private val scanSpeed = 8f

    private var frameCount = 0

    fun updateDetections(list: List<EnemyBox>, w: Int, h: Int) {
        detections = list
        srcW = w
        srcH = h
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val vw = width.toFloat()
        val vh = height.toFloat()
        val scaleX = vw / srcW
        val scaleY = vh / srcH

        // Scan line animation
        scanY += scanSpeed
        if (scanY > vh) scanY = 0f
        canvas.drawRect(0f, scanY, vw, scanY + 3f, scanPaint)

        // Crosshair center
        val cx = vw / 2f
        val cy = vh / 2f
        val aimR = min(vw, vh) * 0.08f
        canvas.drawCircle(cx, cy, aimR, aimCirclePaint)
        canvas.drawLine(cx - aimR * 1.5f, cy, cx - aimR * 0.5f, cy, crosshairPaint)
        canvas.drawLine(cx + aimR * 0.5f, cy, cx + aimR * 1.5f, cy, crosshairPaint)
        canvas.drawLine(cx, cy - aimR * 1.5f, cx, cy - aimR * 0.5f, crosshairPaint)
        canvas.drawLine(cx, cy + aimR * 0.5f, cx, cy + aimR * 1.5f, crosshairPaint)
        canvas.drawCircle(cx, cy, 4f, crosshairPaint.apply { style = Paint.Style.FILL })
        crosshairPaint.style = Paint.Style.STROKE

        // Draw each detection
        for (det in detections) {
            val l = det.left * scaleX
            val t = det.top * scaleY
            val r = det.right * scaleX
            val b = det.bottom * scaleY
            val w = r - l
            val h = b - t

            // Faint fill
            canvas.drawRect(l, t, r, b, fillPaint)

            // Corner brackets (gaming style)
            val cs = min(w, h) * 0.22f
            drawCorners(canvas, l, t, r, b, cs)

            // Confidence-based opacity for full box edge
            boxPaint.alpha = (det.confidence * 180).toInt().coerceIn(80, 200)
            canvas.drawRect(l, t, r, b, boxPaint)

            // Head zone highlight (top 25% of box)
            val headBotY = t + h * 0.28f
            val headPaint = Paint(cornerPaint).apply {
                color = Color.parseColor("#FFFF2233")
                strokeWidth = 2.5f
                style = Paint.Style.STROKE
                alpha = 120
            }
            canvas.drawRect(l + w * 0.2f, t, r - w * 0.2f, headBotY, headPaint)

            // Center vertical line
            val centerLinePaint = Paint(boxPaint).apply {
                alpha = 60
                pathEffect = DashPathEffect(floatArrayOf(6f, 8f), 0f)
                strokeWidth = 1.5f
            }
            canvas.drawLine(l + w / 2f, t, l + w / 2f, b, centerLinePaint)

            // Label background + text
            val conf = (det.confidence * 100).toInt()
            val labelStr = "${det.label}  $conf%"
            val labelW = labelTextPaint.measureText(labelStr) + 20f
            val labelH = 42f
            val labelTop = (t - labelH - 6f).coerceAtLeast(0f)
            val labelRect = RectF(l, labelTop, l + labelW, labelTop + labelH)
            canvas.drawRoundRect(labelRect, 4f, 4f, labelBgPaint)
            canvas.drawText(labelStr, l + 10f, labelTop + 30f, labelTextPaint)

            // Distance estimation line from crosshair to nearest enemy
            if (det == detections.firstOrNull()) {
                val targCx = (l + r) / 2f
                val targCy = (t + b) / 2f
                val linePaint = Paint().apply {
                    color = Color.parseColor("#66FF2233")
                    strokeWidth = 1.5f
                    pathEffect = DashPathEffect(floatArrayOf(10f, 12f), 0f)
                }
                canvas.drawLine(cx, cy, targCx, targCy, linePaint)
            }
        }

        // TOP HUD bar
        canvas.drawRect(0f, 0f, vw, 70f, hudPaint)
        canvas.drawText(
            "AI VISION  ●  ENEMIES: ${detections.size}",
            20f, 50f, hudTextPaint
        )
        frameCount++
        canvas.drawText(
            "ACTIVE",
            vw - 150f, 50f, hudTextPaint
        )

        // Keep animating
        postInvalidateDelayed(80)
    }

    private fun drawCorners(canvas: Canvas, l: Float, t: Float, r: Float, b: Float, cs: Float) {
        // TL
        canvas.drawLine(l, t + cs, l, t, cornerPaint)
        canvas.drawLine(l, t, l + cs, t, cornerPaint)
        // TR
        canvas.drawLine(r - cs, t, r, t, cornerPaint)
        canvas.drawLine(r, t, r, t + cs, cornerPaint)
        // BL
        canvas.drawLine(l, b - cs, l, b, cornerPaint)
        canvas.drawLine(l, b, l + cs, b, cornerPaint)
        // BR
        canvas.drawLine(r - cs, b, r, b, cornerPaint)
        canvas.drawLine(r, b - cs, r, b, cornerPaint)
    }
}
