import 'package:flutter_test/flutter_test.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'package:mushroom_identification/services/storage_service.dart';

void main() {
  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });

  group('StorageService', () {
    late StorageService storageService;

    setUp(() async {
      storageService = StorageService();
      // Close and reset database before each test for isolation
      await storageService.close();
    });

    group('Singleton pattern', () {
      test('StorageService returns same instance', () {
        final instance1 = StorageService();
        final instance2 = StorageService();
        expect(
          identical(instance1, instance2),
          isTrue,
          reason: 'Singleton should return same instance',
        );
      });
    });

    group('Preferences operations', () {
      test('setPreference and getPreference work correctly', () async {
        await storageService.setPreference('test_key', 'test_value');
        final value = await storageService.getPreference('test_key');
        expect(
          value,
          equals('test_value'),
          reason: 'Should retrieve saved preference',
        );
      });

      test('getPreference returns null for non-existent key', () async {
        final value = await storageService.getPreference('non_existent_key');
        expect(
          value,
          isNull,
          reason: 'Should return null for non-existent key',
        );
      });

      test('deletePreference removes preference', () async {
        await storageService.setPreference('delete_test', 'value');
        await storageService.deletePreference('delete_test');
        final value = await storageService.getPreference('delete_test');
        expect(
          value,
          isNull,
          reason: 'Should return null after deletion',
        );
      });

      test('getAllPreferences returns all preferences', () async {
        await storageService.setPreference('pref1', 'value1');
        await storageService.setPreference('pref2', 'value2');
        final prefs = await storageService.getAllPreferences();
        expect(
          prefs.containsKey('pref1'),
          isTrue,
          reason: 'Should contain pref1',
        );
        expect(
          prefs.containsKey('pref2'),
          isTrue,
          reason: 'Should contain pref2',
        );
      });
    });

    group('History entry model', () {
      test('HistoryEntry.toMap includes all fields', () {
        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: {'cap': 'round', 'color': 'brown'},
          results: {'confidence': 0.85, 'species': 'Test Species'},
          createdAt: DateTime(2026, 3, 22),
          notes: 'Test notes',
        );

        final map = entry.toMap();
        expect(map.containsKey('id'), isTrue);
        expect(map.containsKey('imagePath'), isTrue);
        expect(map.containsKey('traits'), isTrue);
        expect(map.containsKey('results'), isTrue);
        expect(map.containsKey('createdAt'), isTrue);
        expect(map.containsKey('notes'), isTrue);
      });

      test('HistoryEntry.fromMap deserializes correctly', () {
        final map = {
          'id': 1,
          'imagePath': '/path/to/image.jpg',
          'traits': '{"cap":"round"}',
          'results': '{"confidence":0.85}',
          'createdAt': '2026-03-22T00:00:00.000Z',
          'notes': 'Test notes',
        };

        final entry = HistoryEntry.fromMap(map);
        expect(entry.id, equals(1));
        expect(entry.imagePath, equals('/path/to/image.jpg'));
        expect(entry.notes, equals('Test notes'));
      });

      test('HistoryEntry.topSpecies returns first prediction', () {
        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: {},
          results: {
            'top_predictions': [
              {'species': 'Amanita muscaria'},
              {'species': 'Amanita caesarea'},
            ]
          },
          createdAt: DateTime.now(),
        );

        expect(
          entry.topSpecies,
          equals('Amanita muscaria'),
          reason: 'Should return first prediction species',
        );
      });

      test('HistoryEntry.confidence returns numeric value', () {
        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: {},
          results: {'confidence': 0.85},
          createdAt: DateTime.now(),
        );

        expect(
          entry.confidence,
          equals(0.85),
          reason: 'Should return confidence as double',
        );
      });

      test('HistoryEntry.safetyRating returns rating', () {
        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: {},
          results: {'safety_rating': 'edible'},
          createdAt: DateTime.now(),
        );

        expect(
          entry.safetyRating,
          equals('edible'),
          reason: 'Should return safety rating',
        );
      });
    });

    group('JSON serialization', () {
      test('Traits are properly JSON encoded/decoded', () async {
        final traits = {
          'cap_shape': 'convex',
          'color': 'brown',
          'gill_type': 'free',
        };

        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: traits,
          results: {},
          createdAt: DateTime.now(),
        );

        final map = entry.toMap();
        final decoded = HistoryEntry.fromMap(map);

        expect(
          decoded.traits,
          equals(traits),
          reason: 'Traits should match after serialization',
        );
      });

      test('Complex results are properly JSON encoded/decoded', () async {
        final results = {
          'confidence': 0.92,
          'method_predictions': {
            'image': {'confidence': 0.90},
            'trait': {'confidence': 0.94},
          },
          'top_predictions': [
            {'species': 'Amanita muscaria', 'confidence': 0.92},
          ],
          'lookalikes': [
            {'species': 'Amanita pantherina', 'risk': 'high'},
          ],
        };

        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: {},
          results: results,
          createdAt: DateTime.now(),
        );

        final map = entry.toMap();
        final decoded = HistoryEntry.fromMap(map);

        expect(
          decoded.results['confidence'],
          equals(0.92),
          reason: 'Confidence should be preserved',
        );
        expect(
          decoded.results['method_predictions'],
          isNotNull,
          reason: 'Method predictions should be preserved',
        );
      });
    });
  });
}
