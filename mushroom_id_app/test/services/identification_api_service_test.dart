import 'dart:typed_data';
import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:mushroom_identification/services/identification_api_service.dart';
import 'package:mushroom_identification/services/storage_service.dart';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

class MockDio extends Mock implements Dio {}

class MockStorageService extends Mock implements StorageService {}

class FakeRequestOptions extends Fake implements RequestOptions {}

void main() {
  setUpAll(() {
    registerFallbackValue(FakeRequestOptions());
  });

  // --------------------------------------------------------------------------
  // IdentificationApiService - unit tests for public methods via mocked Dio
  // --------------------------------------------------------------------------

  group('IdentificationApiService', () {
    late IdentificationApiService service;
    late MockDio mockDio;
    late MockStorageService mockStorage;

    final fakeResponse = <String, dynamic>{
      'species': 'Amanita muscaria',
      'confidence': 0.91,
    };

    setUp(() {
      mockDio = MockDio();
      mockStorage = MockStorageService();
      // Return null so backend auto-detection goes straight to ping
      when(() => mockStorage.getPreference(any()))
          .thenAnswer((_) async => null);
      // Make pings fail → service defaults to Java at port 8080
      when(
        () => mockDio.get<dynamic>(any(), options: any(named: 'options')),
      ).thenThrow(
        DioException(
          requestOptions: RequestOptions(path: '/health'),
          type: DioExceptionType.connectionTimeout,
        ),
      );
      service = IdentificationApiService(
        dio: mockDio,
        storageService: mockStorage,
      );
    });

    // -------------------------------------------------------------------------
    // identifyMushroom
    // -------------------------------------------------------------------------

    group('identifyMushroom', () {
      test('returns parsed response map on success', () async {
        when(
          () => mockDio.post<dynamic>(
            any(),
            data: any(named: 'data'),
            options: any(named: 'options'),
          ),
        ).thenAnswer(
          (_) async => Response(
            requestOptions: RequestOptions(path: '/identify'),
            statusCode: 200,
            data: fakeResponse,
          ),
        );

        final result = await service.identifyMushroom(
          imagePath: 'test.jpg',
          imageBytes: Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0]),
          traits: {'cap_color': 'red'},
        );

        expect(result['species'], equals('Amanita muscaria'));
        expect(result['confidence'], equals(0.91));
      });

      test('throws on DioException', () async {
        when(
          () => mockDio.post<dynamic>(
            any(),
            data: any(named: 'data'),
            options: any(named: 'options'),
          ),
        ).thenThrow(
          DioException(
            requestOptions: RequestOptions(path: '/identify'),
            type: DioExceptionType.connectionTimeout,
          ),
        );

        expect(
          () => service.identifyMushroom(
            imagePath: 'test.jpg',
            imageBytes: Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0]),
            traits: {},
          ),
          throwsA(isA<Exception>()),
        );
      });
    });

    // -------------------------------------------------------------------------
    // step2Start
    // -------------------------------------------------------------------------

    group('step2Start', () {
      test('returns question map on success', () async {
        final fakeQuestion = <String, dynamic>{
          'session_id': 'abc123',
          'question': 'What is the cap shape?',
          'options': ['Convex', 'Flat', 'Umbonate'],
        };

        when(
          () => mockDio.post<dynamic>(
            any(),
            data: any(named: 'data'),
            options: any(named: 'options'),
          ),
        ).thenAnswer(
          (_) async => Response(
            requestOptions: RequestOptions(path: '/step2/start'),
            statusCode: 200,
            data: fakeQuestion,
          ),
        );

        final result = await service.step2Start(
          visibleTraits: {'cap_color': 'red'},
          sessionId: null,
        );

        expect(result['session_id'], equals('abc123'));
        expect(result['question'], isA<String>());
      });
    });

    // -------------------------------------------------------------------------
    // step2Answer
    // -------------------------------------------------------------------------

    group('step2Answer', () {
      test('returns next question or done map', () async {
        final fakeNextQuestion = <String, dynamic>{
          'session_id': 'abc123',
          'done': false,
          'question': 'What is the gill attachment?',
          'options': ['Free', 'Adnate'],
        };

        when(
          () => mockDio.post<dynamic>(
            any(),
            data: any(named: 'data'),
            options: any(named: 'options'),
          ),
        ).thenAnswer(
          (_) async => Response(
            requestOptions: RequestOptions(path: '/step2/answer'),
            statusCode: 200,
            data: fakeNextQuestion,
          ),
        );

        final result = await service.step2Answer(
          sessionId: 'abc123',
          answer: 'Convex',
        );

        expect(result['done'], isFalse);
        expect(result['question'], isA<String>());
      });
    });

    // -------------------------------------------------------------------------
    // step4Finalize
    // -------------------------------------------------------------------------

    group('step4Finalize', () {
      test('returns final identification result', () async {
        final finalResult = <String, dynamic>{
          'final_species': 'Amanita muscaria',
          'confidence': 0.93,
          'edibility': 'toxic',
        };

        when(
          () => mockDio.post<dynamic>(
            any(),
            data: any(named: 'data'),
            options: any(named: 'options'),
          ),
        ).thenAnswer(
          (_) async => Response(
            requestOptions: RequestOptions(path: '/step4/finalize'),
            statusCode: 200,
            data: finalResult,
          ),
        );

        final result = await service.step4Finalize(
          step1Result: {'species': 'Amanita muscaria', 'confidence': 0.9},
          step2Result: {'answered': true},
          step3Result: {'matched': true},
        );

        expect(result['final_species'], equals('Amanita muscaria'));
        expect(result['edibility'], equals('toxic'));
      });
    });
  });
}
