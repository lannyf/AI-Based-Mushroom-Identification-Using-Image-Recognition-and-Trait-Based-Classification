import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:logger/logger.dart';

import 'storage_service.dart';

class IdentificationApiService {
  IdentificationApiService({
    Dio? dio,
    StorageService? storageService,
  })  : _dio = dio ?? Dio(),
        _storageService = storageService ?? StorageService();

  final Dio _dio;
  final StorageService _storageService;
  final Logger _logger = Logger();

  Future<Map<String, dynamic>> identifyMushroom({
    required String imagePath,
    Uint8List? imageBytes,
    required Map<String, dynamic> traits,
  }) async {
    final baseUrl =
        await _storageService.getPreference('api_base_url') ?? 'http://localhost:8000';
    final normalizedBaseUrl = baseUrl.endsWith('/')
        ? baseUrl.substring(0, baseUrl.length - 1)
        : baseUrl;

    final fileName = imagePath.split('/').isNotEmpty ? imagePath.split('/').last : 'upload.jpg';

    final MultipartFile multipartFile;
    if (imageBytes != null) {
      multipartFile = MultipartFile.fromBytes(
        imageBytes,
        filename: fileName,
      );
    } else {
      multipartFile = await MultipartFile.fromFile(
        imagePath,
        filename: fileName,
      );
    }

    final formData = FormData.fromMap({
      'image': multipartFile,
      'traits': jsonEncode(traits),
    });

    try {
      final response = await _dio.post<dynamic>(
        '$normalizedBaseUrl/identify',
        data: formData,
        options: Options(
          contentType: 'multipart/form-data',
          responseType: ResponseType.json,
        ),
      );

      if (response.data is Map<String, dynamic>) {
        return response.data as Map<String, dynamic>;
      }
      if (response.data is Map) {
        return Map<String, dynamic>.from(response.data as Map);
      }
      throw Exception('Unexpected API response format');
    } on DioException catch (e) {
      _logger.e('Identification API error: ${e.response?.data ?? e.message}');
      final detail = e.response?.data;
      throw Exception(
        detail is Map && detail['detail'] != null
            ? detail['detail'].toString()
            : 'Failed to identify mushroom. Check that the backend is running.',
      );
    }
  }
}
