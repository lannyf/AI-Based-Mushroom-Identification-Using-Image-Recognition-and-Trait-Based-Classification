import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/pages/questionnaire_page.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';

void main() {
  group('QuestionnairePage', () {
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

    testWidgets('QuestionnairePage displays correctly',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const QuestionnairePage(),
      ));

      expect(find.text('Mushroom Details'), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
    });

    testWidgets('QuestionnairePage has progress indicator',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const QuestionnairePage(),
      ));

      expect(find.byType(LinearProgressIndicator), findsOneWidget);
    });

    testWidgets('QuestionnairePage displays navigation buttons',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const QuestionnairePage(),
      ));

      await tester.pumpAndSettle();

      // Should have Previous/Next buttons
      expect(find.byIcon(Icons.arrow_back), findsWidgets);
      expect(find.byIcon(Icons.arrow_forward), findsWidgets);
    });

    testWidgets('QuestionnairePage displays page progress',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const QuestionnairePage(),
      ));

      // Should show progress indicator for multi-page form
      expect(find.byType(LinearProgressIndicator), findsOneWidget);
    });

    testWidgets('QuestionnairePage shows questions',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const QuestionnairePage(),
      ));

      // First page should show cap shape question
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Cap Shape') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('QuestionnairePage has submit button on last page',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const QuestionnairePage(),
      ));

      // Navigate to last page
      for (int i = 0; i < 5; i++) {
        await tester.tap(find.byIcon(Icons.arrow_forward));
        await tester.pumpAndSettle();
      }

      // Should show submit button on last page
      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is ElevatedButton &&
              (widget.child as Text?)?.data?.contains('Submit') == true,
        ),
        findsWidgets,
      );
    });
  });
}
