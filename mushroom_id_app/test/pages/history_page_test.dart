import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/pages/history_page.dart';
import 'package:mushroom_identification/providers/history_provider.dart';

void main() {
  group('HistoryPage', () {
    late HistoryProvider historyProvider;

    setUp(() {
      if (Get.isRegistered<HistoryProvider>()) {
        Get.delete<HistoryProvider>();
      }
      historyProvider = Get.put(HistoryProvider());
    });

    tearDown(() {
      Get.delete<HistoryProvider>();
    });

    testWidgets('HistoryPage displays correctly',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const HistoryPage(),
      ));

      expect(find.text('Identification History'), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
    });

    testWidgets('HistoryPage shows empty state initially',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const HistoryPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('No identifications yet') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('HistoryPage shows history icon in empty state',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const HistoryPage(),
      ));

      expect(find.byIcon(Icons.history), findsOneWidget);
    });

    testWidgets('HistoryPage has instructions text',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const HistoryPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Start by taking a photo') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('HistoryPage has Material 3 design',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const HistoryPage(),
      ));

      // Should have a Scaffold with proper structure
      expect(find.byType(Scaffold), findsOneWidget);
      expect(find.byType(ListView), findsOneWidget);
    });
  });
}
