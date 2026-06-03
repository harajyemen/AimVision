package com.aimvision

import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.aimvision.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var mediaProjectionManager: MediaProjectionManager
    private var isServiceRunning = false

    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (Settings.canDrawOverlays(this)) {
            requestMediaProjection()
        } else {
            Toast.makeText(this, "يجب منح إذن الطبقة العلوية", Toast.LENGTH_LONG).show()
        }
    }

    private val mediaProjectionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            startOverlayService(result.resultCode, result.data!!)
        } else {
            Toast.makeText(this, "تم رفض إذن تسجيل الشاشة", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        mediaProjectionManager =
            getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

        binding.btnToggle.setOnClickListener {
            if (isServiceRunning) {
                stopService()
            } else {
                checkPermissionsAndStart()
            }
        }

        binding.btnExit.setOnClickListener {
            stopService()
            finish()
        }
    }

    private fun checkPermissionsAndStart() {
        if (!Settings.canDrawOverlays(this)) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            overlayPermissionLauncher.launch(intent)
        } else {
            requestMediaProjection()
        }
    }

    private fun requestMediaProjection() {
        mediaProjectionLauncher.launch(
            mediaProjectionManager.createScreenCaptureIntent()
        )
    }

    private fun startOverlayService(resultCode: Int, data: Intent) {
        val intent = Intent(this, OverlayService::class.java).apply {
            putExtra(OverlayService.EXTRA_RESULT_CODE, resultCode)
            putExtra(OverlayService.EXTRA_PROJECTION_DATA, data)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        isServiceRunning = true
        updateUI()
        Toast.makeText(this, "تم تشغيل الكشف — يمكنك الآن فتح أي لعبة", Toast.LENGTH_LONG).show()
        // Move app to background so game is visible
        moveTaskToBack(true)
    }

    private fun stopService() {
        stopService(Intent(this, OverlayService::class.java))
        isServiceRunning = false
        updateUI()
    }

    private fun updateUI() {
        if (isServiceRunning) {
            binding.btnToggle.text = "إيقاف الكشف"
            binding.btnToggle.setBackgroundColor(0xFFFF3333.toInt())
            binding.statusText.text = "● نشط — الكشف يعمل"
            binding.statusText.setTextColor(0xFF00FF44.toInt())
        } else {
            binding.btnToggle.text = "تشغيل الكشف"
            binding.btnToggle.setBackgroundColor(0xFF00CC44.toInt())
            binding.statusText.text = "○ متوقف"
            binding.statusText.setTextColor(0xFF888888.toInt())
        }
    }
}
