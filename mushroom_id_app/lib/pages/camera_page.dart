import 'dart:math' show pi;
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';

/// Camera page for capturing or uploading mushroom images.
/// 
/// Allows users to:
/// - Take a photo using device camera
/// - Pick image from gallery
/// - Preview and crop/rotate the image
/// - Confirm or retake the image
/// - Navigate to questionnaire page
class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  final ImagePicker _imagePicker = ImagePicker();
  XFile? _selectedImage;
  Uint8List? _selectedImageBytes;
  double _rotationAngle = 0;
  late TransformationController _transformationController;

  @override
  void initState() {
    super.initState();
    _transformationController = TransformationController();
  }

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  /// Captures image using device camera
  Future<void> _capturePhoto() async {
    try {
      final XFile? photo = await _imagePicker.pickImage(
        source: ImageSource.camera,
        imageQuality: 85,
      );
      if (photo != null) {
        await _setSelectedImage(photo);
      }
    } catch (e) {
      _showErrorSnackBar('Failed to capture photo: $e');
    }
  }

  /// Picks image from device gallery
  Future<void> _pickFromGallery() async {
    try {
      final XFile? image = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 85,
      );
      if (image != null) {
        await _setSelectedImage(image);
      }
    } catch (e) {
      _showErrorSnackBar('Failed to pick image: $e');
    }
  }

  Future<void> _setSelectedImage(XFile image) async {
    final imageBytes = await image.readAsBytes();
    setState(() {
      _selectedImage = image;
      _selectedImageBytes = imageBytes;
      _rotationAngle = 0;
      _transformationController.value = Matrix4.identity();
    });
  }

  /// Rotates image by 90 degrees clockwise
  void _rotateImage() {
    setState(() {
      _rotationAngle = (_rotationAngle + 90) % 360;
    });
  }

  /// Resets image transformation (rotation, zoom, pan)
  void _resetTransformation() {
    setState(() {
      _rotationAngle = 0;
      _transformationController.value = Matrix4.identity();
    });
  }

  /// Resets to initial state (no image selected)
  void _retakePhoto() {
    setState(() {
      _selectedImage = null;
      _selectedImageBytes = null;
      _rotationAngle = 0;
      _transformationController.value = Matrix4.identity();
    });
  }

  /// Validates image and navigates to questionnaire page
  /// 
  /// Validation checks:
  /// - Image is not null
  /// - Image file exists
  /// - File size is within limits (5MB max)
  void _confirmAndContinue() async {
    if (_selectedImage == null) {
      _showErrorSnackBar('No image selected');
      return;
    }

    try {
      final int fileSizeInBytes =
          _selectedImageBytes?.lengthInBytes ?? await _selectedImage!.length();
      const int maxSizeInBytes = 5 * 1024 * 1024; // 5MB

      if (fileSizeInBytes > maxSizeInBytes) {
        _showErrorSnackBar(
          'Image is too large (${(fileSizeInBytes / 1024 / 1024).toStringAsFixed(1)}MB). '
          'Maximum size is 5MB.',
        );
        return;
      }

      // Store selected image in arguments for questionnaire page
      Get.toNamed(
        '/questionnaire',
        arguments: {
          'imagePath': _selectedImage!.path,
          'imageName': _selectedImage!.name,
          'imageBytes': _selectedImageBytes,
        },
      );
    } catch (e) {
      _showErrorSnackBar('Error validating image: $e');
    }
  }

  /// Shows error message to user
  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red[700],
        duration: const Duration(seconds: 3),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = MediaQuery.of(context).size.width < 600;
    final primaryColor = Theme.of(context).primaryColor;
    final secondaryColor = Theme.of(context).colorScheme.secondary;

    if (_selectedImage == null) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Capture Mushroom Image'),
          centerTitle: true,
          elevation: 0,
        ),
        body: Center(
          child: SingleChildScrollView(
            child: Padding(
              padding: EdgeInsets.all(isMobile ? 24.0 : 32.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Large camera icon
                  Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: primaryColor.withOpacity(0.1),
                    ),
                    child: Icon(
                      Icons.camera_alt,
                      size: 60,
                      color: primaryColor,
                    ),
                  ),
                  const SizedBox(height: 32),

                  // Instructions
                  Text(
                    'Capture a Mushroom Image',
                    style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Take a clear photo of the mushroom from above. '
                    'Include the cap, gills (if visible), and stem for best results.',
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                          color: Colors.grey[600],
                        ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 40),

                  // Camera button
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton.icon(
                      onPressed: _capturePhoto,
                      icon: const Icon(Icons.camera_alt, size: 24),
                      label: const Text('Take Photo'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: primaryColor,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),

                  // Gallery button
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: OutlinedButton.icon(
                      onPressed: _pickFromGallery,
                      icon: const Icon(Icons.image, size: 24),
                      label: const Text('Upload from Gallery'),
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(color: primaryColor, width: 2),
                        foregroundColor: primaryColor,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 32),

                  // Help section
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.blue[50],
                      border: Border.all(color: Colors.blue[200]!),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.info_outline, color: Colors.blue[700]),
                            const SizedBox(width: 8),
                            Text(
                              'Photography Tips',
                              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                                    color: Colors.blue[900],
                                    fontWeight: FontWeight.bold,
                                  ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        _buildTipItem(
                          '✓ Use good natural lighting',
                          'Avoid shadows and backlighting',
                        ),
                        const SizedBox(height: 8),
                        _buildTipItem(
                          '✓ Center the mushroom in frame',
                          'Leave some space around edges',
                        ),
                        const SizedBox(height: 8),
                        _buildTipItem(
                          '✓ Capture multiple angles',
                          'Show cap, gills, and stem if possible',
                        ),
                        const SizedBox(height: 8),
                        _buildTipItem(
                          '✓ Keep image sharp',
                          'Hold steady or use tripod',
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    } else {
      // Image preview and editing mode
      return Scaffold(
        appBar: AppBar(
          title: const Text('Preview & Adjust'),
          centerTitle: true,
          elevation: 0,
        ),
        body: Column(
          children: [
            // Image preview with zoom/pan capability
            Expanded(
              child: Container(
                color: Colors.black87,
                child: Center(
                  child: InteractiveViewer(
                     transformationController: _transformationController,
                     minScale: 0.5,
                     maxScale: 4,
                     child: Transform.rotate(
                       angle: _rotationAngle * pi / 180,
                       child: _buildPreviewImage(),
                     ),
                   ),
                 ),
              ),
            ),

            // Image info and adjustment controls
            Container(
              color: Colors.grey[50],
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Image file info
                  Text(
                    'File: ${_selectedImage!.name}',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Colors.grey[600],
                        ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 12),

                  // Adjustment controls
                  Row(
                    children: [
                      // Reset button
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _resetTransformation,
                          icon: const Icon(Icons.refresh),
                          label: const Text('Reset'),
                        ),
                      ),
                      const SizedBox(width: 8),

                      // Rotate button
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _rotateImage,
                          icon: const Icon(Icons.rotate_90_degrees_ccw),
                          label: const Text('Rotate'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),

                  // Action buttons
                  Row(
                    children: [
                      // Retake button
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _retakePhoto,
                          icon: const Icon(Icons.close),
                          label: const Text('Retake'),
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.red[700],
                            side: BorderSide(color: Colors.red[700]!),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),

                      // Continue button
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _confirmAndContinue,
                          icon: const Icon(Icons.check),
                          label: const Text('Continue'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Theme.of(context).primaryColor,
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      );
    }
  }

  /// Builds a single tip item for photography tips section
  Widget _buildTipItem(String title, String description) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          description,
          style: TextStyle(
            fontSize: 13,
            color: Colors.grey[700],
          ),
        ),
      ],
    );
  }

  Widget _buildPreviewImage() {
    if (_selectedImage == null) {
      return const SizedBox.shrink();
    }

    if (kIsWeb) {
      return Image.network(
        _selectedImage!.path,
        errorBuilder: (context, error, stackTrace) {
          if (_selectedImageBytes != null) {
            return Image.memory(_selectedImageBytes!);
          }
          return const Icon(
            Icons.broken_image_outlined,
            color: Colors.white70,
            size: 80,
          );
        },
      );
    }

    if (_selectedImageBytes != null) {
      return Image.memory(_selectedImageBytes!);
    }

    return const SizedBox.shrink();
  }
}
