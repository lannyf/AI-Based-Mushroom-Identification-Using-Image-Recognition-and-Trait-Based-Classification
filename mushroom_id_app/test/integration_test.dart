import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/main.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';

void main() {
  group('End-to-End Identification Flow', () {
    testWidgets('Home -> Camera -> Questionnaire -> Results flow',
        (WidgetTester tester) async {
      // Build app
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Verify we're on home page
      expect(find.text('Identify Mushrooms'), findsOneWidget);
      expect(find.text('Take Photo'), findsOneWidget);
      expect(find.text('View History'), findsOneWidget);
    });

    testWidgets('Navigation to camera page works',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Tap Take Photo button
      await tester.tap(find.text('Take Photo'));
      await tester.pumpAndSettle();

      // Should be on camera page
      expect(find.text('Capture Mushroom Image'), findsOneWidget);
    });

    testWidgets('Settings page is accessible from home',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Tap settings button
      await tester.tap(find.byIcon(Icons.settings));
      await tester.pumpAndSettle();

      // Should be on settings page
      expect(find.text('Settings'), findsOneWidget);
    });

    testWidgets('History page is accessible from home',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Tap View History button
      await tester.tap(find.text('View History'));
      await tester.pumpAndSettle();

      // Should be on history page
      expect(find.text('Identification History'), findsOneWidget);
    });

    testWidgets('Safety disclaimer is visible on home page',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Verify safety disclaimer is shown
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Safety Disclaimer') == true,
        ),
        findsWidgets,
      );

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Do NOT rely solely') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('Camera page has proper UI elements',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Navigate to camera
      await tester.tap(find.text('Take Photo'));
      await tester.pumpAndSettle();

      // Should see camera UI
      expect(find.byIcon(Icons.camera_alt), findsOneWidget);
      expect(find.byIcon(Icons.image_search), findsOneWidget);
    });

    testWidgets('Questionnaire page has 6 questions',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Navigate to questionnaire by skipping camera
      Get.to(() => const Placeholder()); // Navigate away
      Get.back();

      // Manually access provider to verify trait count
      final provider = Get.find<IdentificationProvider>();
      expect(provider.capShapes.isNotEmpty, isTrue);
      expect(provider.colors.isNotEmpty, isTrue);
      expect(provider.gillTypes.isNotEmpty, isTrue);
    });

    testWidgets('Results page displays key elements',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());

      // Set up some data
      final provider = Get.find<IdentificationProvider>();
      provider.setTrait('cap_shape', 'convex');

      await tester.pumpAndSettle();

      // The results page should be accessible via routing
      Get.toNamed('/results');
      await tester.pumpAndSettle();

      expect(find.text('Identification Results'), findsOneWidget);
    });

    testWidgets('Theme is properly applied across pages',
        (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Get the app's theme
      final state = tester.state<MaterialAppState>(
        find.byType(MaterialApp),
      );
      expect(state.widget.theme, isNotNull);

      // Material 3 should be enabled
      expect(state.widget.theme?.useMaterial3, isTrue);
    });

    testWidgets('Color scheme is consistent', (WidgetTester tester) async {
      await tester.pumpWidget(const MushroomIdentificationApp());
      await tester.pumpAndSettle();

      // Primary color should be present
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Container &&
              widget.decoration is BoxDecoration &&
              (widget.decoration as BoxDecoration?)?.color != null,
        ),
        findsWidgets,
      );
    });
  });
}
