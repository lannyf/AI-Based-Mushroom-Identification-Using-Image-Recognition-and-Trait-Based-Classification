import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:logger/logger.dart';

import 'storage_service.dart';

/// API service for the full 4-step mushroom identification pipeline.
///
/// Backend routing (auto-detected at first call):
///   • Java Spring Boot on port 8080  → routes prefixed with /api/
///   • Python FastAPI  on port 8000   → routes without /api/ prefix
///
/// The service probes both backends at startup and selects the first
/// reachable one.  The user-configured url in settings overrides this.
class IdentificationApiService {
  IdentificationApiService({
    Dio? dio,
    StorageService? storageService,
  })  : _dio = dio ?? Dio(),
        _storageService = storageService ?? StorageService();

  final Dio _dio;
  final StorageService _storageService;
  final Logger _logger = Logger();

  // Cached resolved backend info
  String? _resolvedBase;
  String? _resolvedPrefix;  // '/api' for Java, '' for Python

  // -------------------------------------------------------------------------
  // Backend resolution
  // -------------------------------------------------------------------------

  /// Determine which backend to use.
  ///
  /// Priority:
  ///   1. User-configured url from settings (api_base_url key)
  ///   2. Java backend on http://localhost:8080  (prefix /api)
  ///   3. Python backend on http://localhost:8000 (no prefix)
  Future<_Backend> _resolveBackend() async {
    if (_resolvedBase != null) {
      return _Backend(_resolvedBase!, _resolvedPrefix!);
    }

    final stored = await _storageService.getPreference('api_base_url');
    if (stored != null && stored.isNotEmpty) {
      final base = stored.endsWith('/') ? stored.substring(0, stored.length - 1) : stored;
      // Detect prefix: Java backend health is at /api/health
      final prefix = await _ping('$base/api/health') ? '/api'
                   : await _ping('$base/health')     ? ''
                   : '/api'; // assume Java if unknown
      _resolvedBase   = base;
      _resolvedPrefix = prefix;
      return _Backend(base, prefix);
    }

    // Auto-detect: try Java first, then Python
    if (await _ping('http://localhost:8080/api/health')) {
      _logger.i('Backend: Java Spring Boot on port 8080');
      _resolvedBase   = 'http://localhost:8080';
      _resolvedPrefix = '/api';
      return const _Backend('http://localhost:8080', '/api');
    }

    if (await _ping('http://localhost:8000/health')) {
      _logger.i('Backend: Python FastAPI on port 8000');
      _resolvedBase   = 'http://localhost:8000';
      _resolvedPrefix = '';
      return const _Backend('http://localhost:8000', '');
    }

    // Neither reachable — return Java as default so the error clearly
    // identifies the expected endpoint
    _logger.w('No backend reachable; defaulting to Java at port 8080');
    _resolvedBase   = 'http://localhost:8080';
    _resolvedPrefix = '/api';
    return const _Backend('http://localhost:8080', '/api');
  }

  Future<bool> _ping(String url) async {
    try {
      final resp = await _dio.get<dynamic>(
        url,
        options: Options(
          receiveTimeout: const Duration(seconds: 3),
          sendTimeout:    const Duration(seconds: 3),
          validateStatus: (s) => s != null && s < 500,
        ),
      );
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Call this when the user changes the api_base_url in settings
  /// so the cached endpoint is re-resolved next time.
  void invalidateCache() {
    _resolvedBase   = null;
    _resolvedPrefix = null;
  }

  Map<String, dynamic> _toMap(dynamic data) {
    if (data is Map<String, dynamic>) return data;
    if (data is Map) return Map<String, dynamic>.from(data);
    throw Exception('Unexpected API response format');
  }

  // -------------------------------------------------------------------------
  // Step 1 — image upload + visual analysis
  // -------------------------------------------------------------------------

  /// Upload a mushroom image and receive:
  ///   - step1.ml_prediction  : top ML species with confidence
  ///   - step1.visible_traits : colour, cap_shape, texture, ridges
  ///   - top_prediction, predictions, lookalikes
  Future<Map<String, dynamic>> identifyStep1({
    required String imagePath,
    Uint8List? imageBytes,
    Map<String, dynamic> traits = const {},
  }) async {
    final backend = await _resolveBackend();
    final fileName = imagePath.split('/').last.isNotEmpty
        ? imagePath.split('/').last
        : 'upload.jpg';

    final MultipartFile multipartFile = imageBytes != null
        ? MultipartFile.fromBytes(imageBytes, filename: fileName)
        : await MultipartFile.fromFile(imagePath, filename: fileName);

    final formData = FormData.fromMap({
      'image': multipartFile,
      'traits': jsonEncode(traits),
    });

    try {
      final response = await _dio.post<dynamic>(
        '${backend.base}${backend.prefix}/identify',
        data: formData,
        options: Options(contentType: 'multipart/form-data', responseType: ResponseType.json),
      );
      return _toMap(response.data);
    } on DioException catch (e) {
      _logger.e('Step 1 error [${backend.base}]: ${e.response?.data ?? e.message}');
      _rethrow(e, 'Failed to analyse image. Is the backend running at ${backend.base}?');
    }
  }

  // -------------------------------------------------------------------------
  // Step 2 — species key tree traversal
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> step2Start({
    required Map<String, dynamic> visibleTraits,
    String? sessionId,
  }) async {
    final backend = await _resolveBackend();
    try {
      final response = await _dio.post<dynamic>(
        '${backend.base}${backend.prefix}/identify/Species_tree_traversal/start',
        data: {
          if (sessionId != null) 'session_id': sessionId,
          'visible_traits': visibleTraits,
        },
        options: Options(contentType: 'application/json', responseType: ResponseType.json),
      );
      return _toMap(response.data);
    } on DioException catch (e) {
      _logger.e('Step 2 start error: ${e.response?.data ?? e.message}');
      _rethrow(e, 'Failed to start species traversal.');
    }
  }

  Future<Map<String, dynamic>> step2Answer({
    required String sessionId,
    required String answer,
  }) async {
    final backend = await _resolveBackend();
    try {
      final response = await _dio.post<dynamic>(
        '${backend.base}${backend.prefix}/identify/Species_tree_traversal/answer',
        data: {'session_id': sessionId, 'answer': answer},
        options: Options(contentType: 'application/json', responseType: ResponseType.json),
      );
      return _toMap(response.data);
    } on DioException catch (e) {
      _logger.e('Step 2 answer error: ${e.response?.data ?? e.message}');
      _rethrow(e, 'Failed to submit answer.');
    }
  }

  // -------------------------------------------------------------------------
  // Step 3 — trait database comparison
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> step3Compare({
    required String swedishName,
    required Map<String, dynamic> visibleTraits,
  }) async {
    final backend = await _resolveBackend();
    try {
      final response = await _dio.post<dynamic>(
        '${backend.base}${backend.prefix}/identify/comparison/compare',
        data: {'swedish_name': swedishName, 'visible_traits': visibleTraits},
        options: Options(contentType: 'application/json', responseType: ResponseType.json),
      );
      return _toMap(response.data);
    } on DioException catch (e) {
      _logger.e('Step 3 error: ${e.response?.data ?? e.message}');
      _rethrow(e, 'Failed to compare traits.');
    }
  }

  // -------------------------------------------------------------------------
  // Step 4 — final aggregation
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> step4Finalize({
    required Map<String, dynamic> step1Result,
    required Map<String, dynamic> step2Result,
    required Map<String, dynamic> step3Result,
  }) async {
    final backend = await _resolveBackend();
    try {
      final response = await _dio.post<dynamic>(
        '${backend.base}${backend.prefix}/identify/prediction/finalize',
        data: {
          'trait_extraction_result': step1Result,
          'Species_tree_traversal_result': step2Result,
          'comparison_result': step3Result,
        },
        options: Options(contentType: 'application/json', responseType: ResponseType.json),
      );
      return _toMap(response.data);
    } on DioException catch (e) {
      _logger.e('Step 4 error: ${e.response?.data ?? e.message}');
      _rethrow(e, 'Failed to finalise identification.');
    }
  }

  // -------------------------------------------------------------------------
  // Legacy wrapper (kept for compatibility with any existing callers)
  // -------------------------------------------------------------------------

  Future<Map<String, dynamic>> identifyMushroom({
    required String imagePath,
    Uint8List? imageBytes,
    required Map<String, dynamic> traits,
  }) =>
      identifyStep1(imagePath: imagePath, imageBytes: imageBytes, traits: traits);

  // -------------------------------------------------------------------------
  // Helper
  // -------------------------------------------------------------------------

  Never _rethrow(DioException e, String fallbackMessage) {
    // Connection refused / no route to host
    if (e.type == DioExceptionType.connectionError ||
        e.type == DioExceptionType.connectionTimeout ||
        e.type == DioExceptionType.receiveTimeout) {
      throw Exception(
        'Cannot reach backend. Start it with:\n'
        '  make api          (Python on port 8000)\n'
        '  make java-backend (Java  on port 8080)',
      );
    }
    final detail = e.response?.data;
    throw Exception(
      detail is Map && detail['detail'] != null
          ? detail['detail'].toString()
          : fallbackMessage,
    );
  }
}

/// Holds resolved backend base URL and route prefix.
class _Backend {
  const _Backend(this.base, this.prefix);
  final String base;    // e.g. 'http://localhost:8000'
  final String prefix;  // '' for Python, '/api' for Java
}
