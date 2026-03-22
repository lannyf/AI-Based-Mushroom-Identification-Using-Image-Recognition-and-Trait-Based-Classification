import 'dart:io';
import 'package:logger/logger.dart';

/// Service for image handling operations.
/// 
/// Provides functionality for:
/// - Image validation (format, file existence, size)
/// - File size checking and formatting
/// - Image format validation
class ImageService {
  static final ImageService _instance = ImageService._internal();
  final Logger _logger = Logger();

  // Configuration constants
  static const int maxImageSizeInBytes = 5 * 1024 * 1024; // 5MB
  static const int compressionQuality = 85;
  static const List<String> supportedFormats = ['jpg', 'jpeg', 'png'];

  factory ImageService() {
    return _instance;
  }

  ImageService._internal();

  /// Returns true if the file path has a supported image extension.
  /// This is a synchronous, path-only check (no file existence or size check).
  bool isValidExtension(String filePath) {
    final String extension = _getFileExtension(filePath).toLowerCase();
    return supportedFormats.contains(extension);
  }

  /// Formats a raw byte count to a human-readable string (e.g., '1.5 KB').
  String formatFileSize(int bytes) {
    return _formatBytes(bytes);
  }

  /// Returns an error message for a path with an unsupported extension,
  /// or an empty string if the extension is supported.
  /// This is a synchronous, path-only check (no file existence or size check).
  String validateExtension(String filePath) {
    final String extension = _getFileExtension(filePath).toLowerCase();
    if (!supportedFormats.contains(extension)) {
      return 'Unsupported image format ($extension). Supported: ${supportedFormats.join(", ")}';
    }
    return '';
  }

  /// Validates if a file is a valid image
  /// 
  /// Checks:
  /// - File exists
  /// - File extension is supported
  /// - File size is within limits
  /// 
  /// Returns true if valid, false otherwise
  Future<bool> isValidImage(File imageFile) async {
    try {
      if (!await imageFile.exists()) {
        _logger.w('Image file does not exist: ${imageFile.path}');
        return false;
      }

      final String extension = _getFileExtension(imageFile.path).toLowerCase();
      if (!supportedFormats.contains(extension)) {
        _logger.w('Unsupported image format: $extension');
        return false;
      }

      final int fileSizeInBytes = await imageFile.length();
      if (fileSizeInBytes > maxImageSizeInBytes) {
        _logger.w(
          'Image file too large: ${(fileSizeInBytes / 1024 / 1024).toStringAsFixed(2)}MB',
        );
        return false;
      }

      return true;
    } catch (e) {
      _logger.e('Error validating image: $e');
      return false;
    }
  }

  /// Gets the file size in bytes
  Future<int> getFileSizeInBytes(File imageFile) async {
    try {
      return await imageFile.length();
    } catch (e) {
      _logger.e('Error getting file size: $e');
      return -1;
    }
  }

  /// Gets the file size as a formatted string
  /// 
  /// Returns size in appropriate unit (B, KB, MB)
  Future<String> getFileSizeFormatted(File imageFile) async {
    try {
      final int sizeInBytes = await imageFile.length();
      return _formatBytes(sizeInBytes);
    } catch (e) {
      _logger.e('Error formatting file size: $e');
      return 'Unknown';
    }
  }

  /// Gets image dimensions (width, height)
  /// 
  /// Note: For production, use image package to decode image
  /// and get actual dimensions. This is a placeholder.
  Future<Map<String, int>> getImageDimensions(File imageFile) async {
    try {
      // This would require the 'image' package to properly decode
      // For now, return placeholder values
      _logger.i('Image dimensions would be extracted here');
      return {'width': 0, 'height': 0};
    } catch (e) {
      _logger.e('Error getting image dimensions: $e');
      return {'width': 0, 'height': 0};
    }
  }

  /// Validates image file and returns a human-readable error message
  /// 
  /// Returns null if image is valid, error message string otherwise
  Future<String?> validateImageWithErrorMessage(File imageFile) async {
    try {
      if (!await imageFile.exists()) {
        return 'Image file not found';
      }

      final String extension = _getFileExtension(imageFile.path).toLowerCase();
      if (!supportedFormats.contains(extension)) {
        return 'Unsupported image format ($extension). Supported: ${supportedFormats.join(", ")}';
      }

      final int fileSizeInBytes = await imageFile.length();
      if (fileSizeInBytes > maxImageSizeInBytes) {
        final String sizeFormatted = _formatBytes(fileSizeInBytes);
        final String maxFormatted = _formatBytes(maxImageSizeInBytes);
        return 'Image too large ($sizeFormatted). Maximum: $maxFormatted';
      }

      return null; // Valid image
    } catch (e) {
      _logger.e('Error validating image: $e');
      return 'Error validating image: $e';
    }
  }

  /// Extracts file extension from path
  String _getFileExtension(String path) {
    try {
      return path.split('.').last;
    } catch (e) {
      _logger.e('Error extracting file extension: $e');
      return '';
    }
  }

  /// Formats bytes to human-readable string
  String _formatBytes(int bytes) {
    if (bytes < 1024) {
      return '$bytes B';
    } else if (bytes < 1024 * 1024) {
      return '${(bytes / 1024).toStringAsFixed(1)} KB';
    } else {
      return '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB';
    }
  }

  /// Logs image validation details for debugging
  Future<void> logImageDetails(File imageFile) async {
    try {
      final int sizeInBytes = await imageFile.length();
      final String extension = _getFileExtension(imageFile.path);

      _logger.i(
        'Image: ${imageFile.path}, '
        'Size: ${_formatBytes(sizeInBytes)}, '
        'Format: $extension',
      );
    } catch (e) {
      _logger.e('Error logging image details: $e');
    }
  }
}
