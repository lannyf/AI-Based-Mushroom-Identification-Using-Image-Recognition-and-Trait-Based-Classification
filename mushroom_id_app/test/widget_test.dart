// App-level smoke tests — verifies the real MushroomIdentificationApp renders
// without crashing and shows the home screen.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/main.dart';
import 'package:mushroom_identification/providers/history_provider.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';
import 'package:mushroom_identification/providers/language_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });

  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    Get.reset();
    // Register providers as main() would — must happen before pumpWidget
    Get.put(LanguageProvider());
    Get.put(IdentificationProvider());
    Get.put(HistoryProvider());
    await Future.delayed(Duration.zero);
  });

  tearDown(() {
    Get.reset();
  });

  testWidgets('App renders without crashing', (WidgetTester tester) async {
    await tester.pumpWidget(const MushroomIdentificationApp());
    await tester.pump();
    expect(find.byType(MushroomIdentificationApp), findsOneWidget);
  });

  testWidgets('Home screen shows text content', (WidgetTester tester) async {
    await tester.pumpWidget(const MushroomIdentificationApp());
    await tester.pump();
    expect(
      find.byWidgetPredicate(
        (widget) =>
            widget is Text &&
            (widget.data?.toLowerCase().contains('mushroom') == true ||
                widget.data?.toLowerCase().contains('identify') == true ||
                widget.data?.toLowerCase().contains('svamp') == true),
      ),
      findsWidgets,
    );
  });

  testWidgets('App has at least one AppBar', (WidgetTester tester) async {
    await tester.pumpWidget(const MushroomIdentificationApp());
    await tester.pump();
    expect(find.byType(AppBar), findsWidgets);
  });
}
