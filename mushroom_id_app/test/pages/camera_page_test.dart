import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/main.dart';
import 'package:mushroom_identification/pages/camera_page.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';

void main() {
  group('CameraPage', () {
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

    testWidgets('CameraPage displays selection mode initially',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const CameraPage(),
        getPages: [
          GetPage(name: '/questionnaire', page: () => Container()),
        ],
      ));

      // Initial state should show selection buttons
      expect(find.text('Take Photo'), findsOneWidget);
      expect(find.text('Upload from Gallery'), findsOneWidget);
    });

    testWidgets('CameraPage has proper app bar', (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const CameraPage(),
      ));

      expect(find.text('Capture Mushroom Image'), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
    });

    testWidgets('CameraPage displays icon buttons', (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const CameraPage(),
      ));

      expect(find.byIcon(Icons.camera_alt), findsOneWidget);
      expect(find.byIcon(Icons.image), findsOneWidget);
    });

    testWidgets('Camera page has informational text',
        (WidgetTester tester) async {
      await tester.pumpWidget(GetMaterialApp(
        home: const CameraPage(),
      ));

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('best quality') == true,
        ),
        findsWidgets,
      );
    });
  });
}
