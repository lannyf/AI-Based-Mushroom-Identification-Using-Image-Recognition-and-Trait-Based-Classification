import 'dart:convert';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'package:logger/logger.dart';

class HistoryEntry {
  final int? id;
  final String imagePath;
  final Map<String, dynamic> traits;
  final Map<String, dynamic> results;
  final DateTime createdAt;
  final String? notes;

  HistoryEntry({
    this.id,
    required this.imagePath,
    required this.traits,
    required this.results,
    required this.createdAt,
    this.notes,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'imagePath': imagePath,
      'traits': jsonEncode(traits),
      'results': jsonEncode(results),
      'createdAt': createdAt.toIso8601String(),
      'notes': notes,
    };
  }

  factory HistoryEntry.fromMap(Map<String, dynamic> map) {
    return HistoryEntry(
      id: map['id'] as int,
      imagePath: map['imagePath'] as String,
      traits: jsonDecode(map['traits'] as String) as Map<String, dynamic>,
      results: jsonDecode(map['results'] as String) as Map<String, dynamic>,
      createdAt: DateTime.parse(map['createdAt'] as String),
      notes: map['notes'] as String?,
    );
  }

  String get topSpecies {
    if (results['top_predictions'] is List && (results['top_predictions'] as List).isNotEmpty) {
      final predictions = results['top_predictions'] as List;
      if (predictions.isNotEmpty && predictions[0] is Map) {
        return (predictions[0] as Map)['species'] as String? ?? 'Unknown';
      }
    }
    return 'Unknown';
  }

  double get confidence {
    final conf = results['confidence'];
    if (conf == null) {
      return 0.0;
    }
    if (conf is double) {
      return conf;
    }
    if (conf is int) {
      return conf.toDouble();
    }
    if (conf is num) {
      return conf.toDouble();
    }
    if (conf is String) {
      final parsed = double.tryParse(conf);
      if (parsed != null) {
        return parsed;
      }
    }
    return 0.0;
  }

  String? get safetyRating {
    return results['safety_rating'] as String?;
  }
}

class StorageService {
  static final StorageService _instance = StorageService._internal();
  static Database? _database;
  final _logger = Logger();

  StorageService._internal();

  factory StorageService() {
    return _instance;
  }

  Future<Database> get database async {
    _database ??= await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    try {
      final databasesPath = await getDatabasesPath();
      final path = join(databasesPath, 'mushroom_identification.db');

      _logger.i('Initializing database at: $path');

      return await openDatabase(
        path,
        version: 1,
        onCreate: _onCreate,
      );
    } catch (e) {
      _logger.e('Database initialization error: $e');
      rethrow;
    }
  }

  Future<void> _onCreate(Database db, int version) async {
    try {
      await db.execute(
        '''
        CREATE TABLE history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          imagePath TEXT NOT NULL,
          traits TEXT NOT NULL,
          results TEXT NOT NULL,
          createdAt TEXT NOT NULL,
          notes TEXT
        )
        ''',
      );
      _logger.i('History table created successfully');

      await db.execute(
        '''
        CREATE TABLE preferences (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        ''',
      );
      _logger.i('Preferences table created successfully');
    } catch (e) {
      _logger.e('Database creation error: $e');
      rethrow;
    }
  }

  // History Operations
  Future<int> saveIdentification(HistoryEntry entry) async {
    try {
      final db = await database;
      final id = await db.insert(
        'history',
        {
          'imagePath': entry.imagePath,
          'traits': jsonEncode(entry.traits),
          'results': jsonEncode(entry.results),
          'createdAt': entry.createdAt.toIso8601String(),
          'notes': entry.notes,
        },
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
      _logger.i('Identification saved with id: $id');
      return id;
    } catch (e) {
      _logger.e('Error saving identification: $e');
      rethrow;
    }
  }

  Future<List<HistoryEntry>> getHistory() async {
    try {
      final db = await database;
      final maps = await db.query(
        'history',
        orderBy: 'createdAt DESC',
      );

      return List.generate(maps.length, (i) {
        return HistoryEntry.fromMap(maps[i]);
      });
    } catch (e) {
      _logger.e('Error loading history: $e');
      rethrow;
    }
  }

  Future<HistoryEntry?> getHistoryEntry(int id) async {
    try {
      final db = await database;
      final maps = await db.query(
        'history',
        where: 'id = ?',
        whereArgs: [id],
      );

      if (maps.isEmpty) return null;
      return HistoryEntry.fromMap(maps.first);
    } catch (e) {
      _logger.e('Error loading history entry: $e');
      rethrow;
    }
  }

  Future<void> deleteHistoryEntry(int id) async {
    try {
      final db = await database;
      await db.delete(
        'history',
        where: 'id = ?',
        whereArgs: [id],
      );
      _logger.i('History entry deleted: $id');
    } catch (e) {
      _logger.e('Error deleting history entry: $e');
      rethrow;
    }
  }

  Future<void> clearHistory() async {
    try {
      final db = await database;
      await db.delete('history');
      _logger.i('History cleared');
    } catch (e) {
      _logger.e('Error clearing history: $e');
      rethrow;
    }
  }

  // Preferences Operations
  Future<void> setPreference(String key, String value) async {
    try {
      final db = await database;
      await db.insert(
        'preferences',
        {'key': key, 'value': value},
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
      _logger.i('Preference saved: $key');
    } catch (e) {
      _logger.e('Error saving preference: $e');
      rethrow;
    }
  }

  Future<String?> getPreference(String key) async {
    try {
      final db = await database;
      final maps = await db.query(
        'preferences',
        where: 'key = ?',
        whereArgs: [key],
      );

      if (maps.isEmpty) return null;
      return maps.first['value'] as String?;
    } catch (e) {
      _logger.e('Error loading preference: $e');
      rethrow;
    }
  }

  Future<Map<String, String>> getAllPreferences() async {
    try {
      final db = await database;
      final maps = await db.query('preferences');

      return {for (var map in maps) map['key'] as String: map['value'] as String};
    } catch (e) {
      _logger.e('Error loading preferences: $e');
      rethrow;
    }
  }

  Future<void> deletePreference(String key) async {
    try {
      final db = await database;
      await db.delete(
        'preferences',
        where: 'key = ?',
        whereArgs: [key],
      );
      _logger.i('Preference deleted: $key');
    } catch (e) {
      _logger.e('Error deleting preference: $e');
      rethrow;
    }
  }

  Future<void> close() async {
    if (_database != null) {
      await _database!.close();
      _database = null;
    }
  }
}
