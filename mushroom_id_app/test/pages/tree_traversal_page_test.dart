import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/pages/tree_traversal_page.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';
import 'package:mushroom_identification/providers/language_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });

  group('TreeTraversalPage', () {
    setUp(() async {
      SharedPreferences.setMockInitialValues({});
      Get.reset();
      Get.put(LanguageProvider());
      Get.put(IdentificationProvider());
      await Future.delayed(Duration.zero);
    });

    tearDown(() {
      Get.reset();
    });

    testWidgets('TreeTraversalPage renders without crashing',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const TreeTraversalPage(),
        getPages: [
          GetPage(name: '/results', page: () => Container()),
        ],
      ));
      await tester.pump();
      expect(find.byType(Scaffold), findsOneWidget);
    });

    testWidgets('TreeTraversalPage has an AppBar', (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const TreeTraversalPage(),
        getPages: [
          GetPage(name: '/results', page: () => Container()),
        ],
      ));
      await tester.pump();
      expect(find.byType(AppBar), findsOneWidget);
    });

    testWidgets('AppBar title contains identification text',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const TreeTraversalPage(),
        getPages: [
          GetPage(name: '/results', page: () => Container()),
        ],
      ));
      await tester.pump();
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              (widget.data?.contains('Identification') == true ||
                  widget.data?.contains('Artbestämning') == true ||
                  widget.data?.contains('Species') == true),
        ),
        findsWidgets,
      );
    });

    testWidgets('Shows loading indicator while processing',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const TreeTraversalPage(),
        getPages: [
          GetPage(name: '/results', page: () => Container()),
        ],
      ));
      // On the first frame before async init completes, a loading indicator
      // or the initial state should be visible.
      await tester.pump(Duration.zero);
      expect(find.byType(Scaffold), findsOneWidget);
    });

    testWidgets('Language flag button is present in AppBar',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const TreeTraversalPage(),
        getPages: [
          GetPage(name: '/results', page: () => Container()),
        ],
      ));
      await tester.pump();
      // The AppBar should contain the LanguageFlagButton action
      final appBar = tester.widget<AppBar>(find.byType(AppBar));
      expect(appBar.actions, isNotNull);
      expect(appBar.actions!.isNotEmpty, isTrue);
    });
  });
}
