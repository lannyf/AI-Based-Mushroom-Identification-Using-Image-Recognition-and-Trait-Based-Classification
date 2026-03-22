import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/pages/results_page.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';

void main() {
  group('ResultsPage', () {
    late IdentificationProvider idProvider;

    setUp(() {
      if (Get.isRegistered<IdentificationProvider>()) {
        Get.delete<IdentificationProvider>();
      }
      idProvider = Get.put(IdentificationProvider());
    });

    tearDown(() {
      Get.delete<IdentificationProvider>();
    });

    testWidgets('ResultsPage displays correctly',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      expect(find.text('Identification Results'), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
    });

    testWidgets('ResultsPage displays confidence indicator',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    testWidgets('ResultsPage displays confidence percentage',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      // Should display confidence percentage
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              (widget.data?.contains('%') == true ||
                  widget.data?.contains('85') == true),
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays predictions section',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Top Predictions') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays method breakdown',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              (widget.data?.contains('Image') == true ||
                  widget.data?.contains('Trait') == true ||
                  widget.data?.contains('LLM') == true),
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays lookalike warnings',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Lookalike') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage displays safety warning',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Safety') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('ResultsPage has action buttons',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      // AppBar has bookmark (save) and share icons; bottom has Share icon
      expect(find.byIcon(Icons.bookmark), findsWidgets);
      expect(find.byIcon(Icons.share), findsWidgets);
    });

    testWidgets('ResultsPage displays disclaimer',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const ResultsPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Important Safety Notice') == true,
        ),
        findsWidgets,
      );
    });
  });
}
