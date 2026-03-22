import 'package:get/get.dart';
import 'package:logger/logger.dart';
import '../services/storage_service.dart';

class HistoryProvider extends GetxController {
  final StorageService _storageService = StorageService();
  final _logger = Logger();

  // Observable state
  final RxList<HistoryEntry> history = <HistoryEntry>[].obs;
  final RxBool isLoading = false.obs;
  final RxString errorMessage = ''.obs;
  final Rx<HistoryEntry?> selectedEntry = Rx<HistoryEntry?>(null);

  @override
  void onInit() {
    super.onInit();
    _logger.i('HistoryProvider initialized');
    loadHistory();
  }

  /// Load all history entries from storage
  Future<void> loadHistory() async {
    try {
      isLoading.value = true;
      errorMessage.value = '';

      final entries = await _storageService.getHistory();
      history.assignAll(entries);
      _logger.i('Loaded ${entries.length} history entries');
    } catch (e) {
      _logger.e('Error loading history: $e');
      errorMessage.value = 'Failed to load history: ${e.toString()}';
      history.clear();
    } finally {
      isLoading.value = false;
    }
  }

  /// Save a new identification result to history
  Future<void> saveIdentification(HistoryEntry entry) async {
    try {
      final id = await _storageService.saveIdentification(entry);
      _logger.i('Identification saved with id: $id');

      // Insert at beginning of list (most recent first)
      history.insert(0, HistoryEntry(
        id: id,
        imagePath: entry.imagePath,
        traits: entry.traits,
        results: entry.results,
        createdAt: entry.createdAt,
        notes: entry.notes,
      ));
      errorMessage.value = '';
    } catch (e) {
      _logger.e('Error saving identification: $e');
      errorMessage.value = 'Failed to save identification: ${e.toString()}';
      rethrow;
    }
  }

  /// Get a specific history entry by ID
  Future<HistoryEntry?> getEntry(int id) async {
    try {
      return await _storageService.getHistoryEntry(id);
    } catch (e) {
      _logger.e('Error getting history entry: $e');
      errorMessage.value = 'Failed to load entry: ${e.toString()}';
      return null;
    }
  }

  /// Select a history entry for detail view
  void selectEntry(HistoryEntry entry) {
    selectedEntry.value = entry;
    _logger.i('Entry selected: ${entry.id}');
  }

  /// Deselect current entry
  void clearSelection() {
    selectedEntry.value = null;
  }

  /// Delete a history entry
  Future<void> deleteEntry(int id) async {
    try {
      await _storageService.deleteHistoryEntry(id);
      history.removeWhere((entry) => entry.id == id);
      if (selectedEntry.value?.id == id) {
        clearSelection();
      }
      _logger.i('Entry deleted: $id');
      errorMessage.value = '';
    } catch (e) {
      _logger.e('Error deleting entry: $e');
      errorMessage.value = 'Failed to delete entry: ${e.toString()}';
      rethrow;
    }
  }

  /// Clear all history
  Future<void> clearAllHistory() async {
    try {
      await _storageService.clearHistory();
      history.clear();
      clearSelection();
      _logger.i('All history cleared');
      errorMessage.value = '';
    } catch (e) {
      _logger.e('Error clearing history: $e');
      errorMessage.value = 'Failed to clear history: ${e.toString()}';
      rethrow;
    }
  }

  /// Get count of history entries
  int get historyCount => history.length;

  /// Check if history is empty
  bool get isEmpty => history.isEmpty;

  /// Get the most recent entry
  HistoryEntry? get mostRecent => history.isNotEmpty ? history.first : null;

  /// Get entries from the past N days
  List<HistoryEntry> getRecentEntries(int days) {
    final now = DateTime.now();
    final cutoff = now.subtract(Duration(days: days));
    return history.where((entry) => entry.createdAt.isAfter(cutoff)).toList();
  }

  /// Search history by species name
  List<HistoryEntry> searchBySpecies(String query) {
    final lowerQuery = query.toLowerCase();
    return history.where((entry) {
      final species = entry.topSpecies.toLowerCase();
      return species.contains(lowerQuery);
    }).toList();
  }

  /// Get average confidence from history
  double getAverageConfidence() {
    if (history.isEmpty) return 0.0;
    final total = history.fold<double>(0.0, (sum, entry) => sum + entry.confidence);
    return total / history.length;
  }

  /// Get safety distribution (edible, caution, inedible)
  Map<String, int> getSafetyDistribution() {
    final distribution = {'edible': 0, 'caution': 0, 'inedible': 0, 'unknown': 0};

    for (final entry in history) {
      final safety = entry.safetyRating ?? 'unknown';
      if (distribution.containsKey(safety)) {
        distribution[safety] = (distribution[safety] ?? 0) + 1;
      }
    }

    return distribution;
  }

  @override
  void onClose() {
    _logger.i('HistoryProvider closed');
    super.onClose();
  }
}
