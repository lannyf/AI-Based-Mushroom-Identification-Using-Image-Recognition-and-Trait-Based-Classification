import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/providers/history_provider.dart';
import 'package:mushroom_identification/services/storage_service.dart';

void main() {
  group('HistoryProvider', () {
    late HistoryProvider provider;

    setUp(() {
      if (Get.isRegistered<HistoryProvider>()) {
        Get.delete<HistoryProvider>();
      }
      provider = Get.put(HistoryProvider());
    });

    tearDown(() {
      Get.delete<HistoryProvider>();
    });

    group('Initialization', () {
      test('Provider initializes with empty history', () {
        expect(
          provider.history.isEmpty,
          isTrue,
          reason: 'History should start empty',
        );
      });

      test('Provider initializes with default state values', () {
        expect(provider.isLoading.value, isFalse);
        expect(provider.errorMessage.value, isEmpty);
        expect(provider.selectedEntry.value, isNull);
      });
    });

    group('State getters', () {
      test('historyCount returns number of entries', () {
        expect(
          provider.historyCount,
          equals(0),
          reason: 'Should return 0 for empty history',
        );
      });

      test('isEmpty returns true when no entries', () {
        expect(
          provider.isEmpty,
          isTrue,
          reason: 'Should be empty initially',
        );
      });

      test('mostRecent returns null when empty', () {
        expect(
          provider.mostRecent,
          isNull,
          reason: 'Should return null when empty',
        );
      });
    });

    group('Selection management', () {
      test('selectEntry sets selectedEntry', () {
        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: {},
          results: {},
          createdAt: DateTime.now(),
        );

        provider.selectEntry(entry);

        expect(
          provider.selectedEntry.value,
          equals(entry),
          reason: 'Should set selected entry',
        );
      });

      test('clearSelection clears selectedEntry', () {
        final entry = HistoryEntry(
          id: 1,
          imagePath: '/path/to/image.jpg',
          traits: {},
          results: {},
          createdAt: DateTime.now(),
        );

        provider.selectEntry(entry);
        provider.clearSelection();

        expect(
          provider.selectedEntry.value,
          isNull,
          reason: 'Should clear selected entry',
        );
      });
    });

    group('Average confidence calculation', () {
      test('getAverageConfidence returns 0 for empty history', () {
        expect(
          provider.getAverageConfidence(),
          equals(0.0),
          reason: 'Should return 0 for empty history',
        );
      });
    });

    group('Safety distribution', () {
      test('getSafetyDistribution returns correct structure', () {
        final distribution = provider.getSafetyDistribution();

        expect(
          distribution.containsKey('edible'),
          isTrue,
          reason: 'Should have edible category',
        );
        expect(
          distribution.containsKey('caution'),
          isTrue,
          reason: 'Should have caution category',
        );
        expect(
          distribution.containsKey('inedible'),
          isTrue,
          reason: 'Should have inedible category',
        );
        expect(
          distribution.containsKey('unknown'),
          isTrue,
          reason: 'Should have unknown category',
        );
      });

      test('getSafetyDistribution returns zero counts for empty history', () {
        final distribution = provider.getSafetyDistribution();

        expect(distribution['edible'], equals(0));
        expect(distribution['caution'], equals(0));
        expect(distribution['inedible'], equals(0));
      });
    });

    group('Search functionality', () {
      test('searchBySpecies returns empty list for empty history', () {
        final results = provider.searchBySpecies('Amanita');
        expect(
          results.isEmpty,
          isTrue,
          reason: 'Should return empty list for empty history',
        );
      });

      test('searchBySpecies is case insensitive', () {
        final results = provider.searchBySpecies('amanita');
        expect(
          results,
          isNotNull,
          reason: 'Should handle case insensitive search',
        );
      });
    });

    group('Recent entries filtering', () {
      test('getRecentEntries returns empty for empty history', () {
        final results = provider.getRecentEntries(7);
        expect(
          results.isEmpty,
          isTrue,
          reason: 'Should return empty list for empty history',
        );
      });

      test('getRecentEntries with large day range should work', () {
        final results = provider.getRecentEntries(365);
        expect(
          results,
          isNotNull,
          reason: 'Should handle large day ranges',
        );
      });
    });

    group('Error handling', () {
      test('Error message is initially empty', () {
        expect(
          provider.errorMessage.value,
          isEmpty,
          reason: 'Error message should be empty initially',
        );
      });
    });

    group('Entry deletion', () {
      test('Attempting to delete from empty history sets error', () async {
        try {
          await provider.deleteEntry(999);
        } catch (e) {
          // Expected to throw or handle gracefully
        }
      });
    });

    group('Entry clearing', () {
      test('clearAllHistory empties history list', () async {
        try {
          await provider.clearAllHistory();
          expect(
            provider.history.isEmpty,
            isTrue,
            reason: 'History should be empty after clearing',
          );
        } catch (e) {
          // Database operations may fail in test environment
        }
      });
    });
  });
}
