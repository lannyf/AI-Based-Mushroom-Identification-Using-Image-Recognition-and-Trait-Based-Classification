import 'package:flutter_test/flutter_test.dart';
import 'package:mushroom_identification/services/image_service.dart';

void main() {
  group('ImageService', () {
    late ImageService imageService;

    setUp(() {
      imageService = ImageService();
    });

    group('Extension validation', () {
      test('isValidExtension returns true for valid file extension', () {
        // Test JPG
        expect(
          imageService.isValidExtension('photo.jpg'),
          equals(true),
          reason: 'Should accept .jpg files',
        );

        // Test JPEG
        expect(
          imageService.isValidExtension('photo.jpeg'),
          equals(true),
          reason: 'Should accept .jpeg files',
        );

        // Test PNG
        expect(
          imageService.isValidExtension('photo.png'),
          equals(true),
          reason: 'Should accept .png files',
        );
      });

      test('isValidExtension returns false for invalid extension', () {
        expect(
          imageService.isValidExtension('photo.gif'),
          equals(false),
          reason: 'Should reject .gif files',
        );

        expect(
          imageService.isValidExtension('photo.txt'),
          equals(false),
          reason: 'Should reject .txt files',
        );

        expect(
          imageService.isValidExtension('no_extension'),
          equals(false),
          reason: 'Should reject files without extension',
        );
      });

      test('isValidExtension is case insensitive', () {
        expect(
          imageService.isValidExtension('photo.JPG'),
          equals(true),
          reason: 'Should accept uppercase JPG',
        );

        expect(
          imageService.isValidExtension('photo.Png'),
          equals(true),
          reason: 'Should accept mixed case PNG',
        );
      });
    });

    group('File size formatting', () {
      test('formatFileSize handles bytes correctly', () {
        // 512 bytes
        expect(
          imageService.formatFileSize(512),
          contains('B'),
          reason: 'Should show bytes for small sizes',
        );

        // 1 KB = 1024 bytes
        expect(
          imageService.formatFileSize(1024),
          contains('KB'),
          reason: 'Should show KB for kilobytes',
        );

        // 1 MB = 1024 * 1024 bytes
        expect(
          imageService.formatFileSize(1024 * 1024),
          contains('MB'),
          reason: 'Should show MB for megabytes',
        );
      });

      test('formatFileSize shows decimal places', () {
        final formatted = imageService.formatFileSize(1536); // 1.5 KB
        expect(
          formatted,
          matches(RegExp(r'\d+\.\d+')),
          reason: 'Should include decimal places',
        );
      });
    });

    group('Configuration constants', () {
      test('maxImageSizeInBytes is 5MB', () {
        expect(
          ImageService.maxImageSizeInBytes,
          equals(5 * 1024 * 1024),
          reason: 'Max size should be 5MB',
        );
      });

      test('supportedFormats includes jpg, jpeg, png', () {
        final formats = ImageService.supportedFormats;
        expect(formats.contains('jpg'), isTrue);
        expect(formats.contains('jpeg'), isTrue);
        expect(formats.contains('png'), isTrue);
      });

      test('supportedFormats has exactly 3 formats', () {
        expect(
          ImageService.supportedFormats.length,
          equals(3),
          reason: 'Should support exactly 3 formats',
        );
      });
    });

    group('Error handling', () {
      test('validateExtension returns appropriate error for invalid format', () {
        final result = imageService.validateExtension('photo.bmp');
        expect(
          result,
          isNotEmpty,
          reason: 'Should return error message for invalid format',
        );
        expect(
          result.toLowerCase(),
          contains('format'),
          reason: 'Error message should mention format',
        );
      });

      test('validateExtension returns empty for valid format', () {
        final result = imageService.validateExtension('photo.jpg');
        expect(
          result,
          isEmpty,
          reason: 'Should return empty string for valid format',
        );
      });
    });
  });
}
