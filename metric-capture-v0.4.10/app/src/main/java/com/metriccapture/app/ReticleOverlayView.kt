package com.metriccapture.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/** Transparent center reticle drawn independently over the continuous camera surface. */
class ReticleOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    var showReticle: Boolean = true
        set(value) {
            field = value
            invalidate()
        }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(220, 255, 200, 0)
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }

    override fun onDraw(canvas: Canvas) {
        if (!showReticle) return
        val x = width * 0.5f
        val y = height * 0.5f
        canvas.drawCircle(x, y, 18f, paint)
        canvas.drawLine(x - 32f, y, x + 32f, y, paint)
        canvas.drawLine(x, y - 32f, x, y + 32f, paint)
    }
}
