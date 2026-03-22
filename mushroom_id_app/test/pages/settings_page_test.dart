import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mushroom_identification/pages/settings_page.dart';

void main() {
  group('SettingsPage', () {
    testWidgets('SettingsPage displays correctly',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      expect(find.text('Settings'), findsOneWidget);
      expect(find.byType(AppBar), findsOneWidget);
    });

    testWidgets('SettingsPage displays app settings section',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('App Settings') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays notifications setting',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Enable Notifications') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays language setting',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Language') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays API configuration section',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('API Configuration') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays storage section',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Storage & Data') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays about section',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('About') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays clear cache option',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Clear Cache') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays privacy policy option',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('Privacy Policy') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage displays app version',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      await tester.pumpAndSettle();

      expect(
        find.byWidgetPredicate(
          (widget) =>
              widget is Text &&
              widget.data?.contains('App Version') == true,
        ),
        findsWidgets,
      );
    });

    testWidgets('SettingsPage is scrollable', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      expect(find.byType(SingleChildScrollView), findsOneWidget);
    });

    testWidgets('SettingsPage has Material 3 design',
        (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: SettingsPage(),
        ),
      );

      // Should have Cards for grouping settings
      expect(find.byType(Card), findsWidgets);
    });
  });
}
