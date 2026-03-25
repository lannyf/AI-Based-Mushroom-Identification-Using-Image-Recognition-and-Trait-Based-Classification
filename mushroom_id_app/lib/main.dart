import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:logger/logger.dart';

// Import pages and providers
import 'pages/camera_page.dart';
import 'pages/questionnaire_page.dart';
import 'pages/results_page.dart';
import 'pages/history_page.dart';
import 'pages/settings_page.dart';
import 'providers/identification_provider.dart';
import 'providers/history_provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize logging
  final logger = Logger();
  logger.i('Mushroom Identification App - Starting');

  // Register GetX controllers once before runApp to avoid re-registration
  // on widget rebuilds.
  Get.put(IdentificationProvider());
  Get.put(HistoryProvider());

  runApp(const MushroomIdentificationApp());
}

class MushroomIdentificationApp extends StatelessWidget {
  const MushroomIdentificationApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Define color scheme
    const primaryColor = Color(0xFF4CAF50);    // Green (trust, nature, safety)
    const secondaryColor = Color(0xFFFF9800);   // Orange (caution, warnings)
    const dangerColor = Color(0xFFF44336);      // Red (toxic species alerts)
    const backgroundColor = Color(0xFFF5F5F5);  // Light gray
    const textColor = Color(0xFF333333);        // Dark gray

    // Create custom theme
    final ThemeData appTheme = ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: primaryColor,
        primary: primaryColor,
        secondary: secondaryColor,
        error: dangerColor,
        brightness: Brightness.light,
      ),
      
      // Text theme with Google Fonts
      textTheme: GoogleFonts.interTextTheme(
        Theme.of(context).textTheme.apply(
          bodyColor: textColor,
          displayColor: textColor,
        ),
      ),
      
      // App bar styling
      appBarTheme: AppBarTheme(
        backgroundColor: primaryColor,
        foregroundColor: Colors.white,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: GoogleFonts.inter(
          fontSize: 20,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
      ),

      // Button styling
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          textStyle: GoogleFonts.inter(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),

      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          side: const BorderSide(color: primaryColor, width: 2),
          foregroundColor: primaryColor,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          textStyle: GoogleFonts.inter(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),

      // Card styling
      cardTheme: CardThemeData(
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        color: Colors.white,
      ),

      // Input decoration for text fields
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.grey[100],
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.grey[300]!),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: Colors.grey[300]!),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: primaryColor, width: 2),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        hintStyle: GoogleFonts.inter(color: Colors.grey[500]),
      ),

      // Dialog theme
      dialogTheme: DialogThemeData(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        elevation: 8,
      ),

      // Floating action button
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: primaryColor,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),

      // Scaffold background
      scaffoldBackgroundColor: backgroundColor,
    );

    return GetMaterialApp(
      title: 'Mushroom Identification',
      theme: appTheme,
      darkTheme: appTheme, // Can customize dark theme later
      themeMode: ThemeMode.light,
      home: const HomePage(),
      getPages: [
        GetPage(
          name: '/',
          page: () => const HomePage(),
        ),
        GetPage(
          name: '/camera',
          page: () => const CameraPage(),
        ),
        GetPage(
          name: '/questionnaire',
          page: () => const QuestionnairePage(),
        ),
        GetPage(
          name: '/results',
          page: () => const ResultsPage(),
        ),
        GetPage(
          name: '/history',
          page: () => const HistoryPage(),
        ),
        GetPage(
          name: '/settings',
          page: () => const SettingsPage(),
        ),
      ],
      debugShowCheckedModeBanner: false,
    );
  }
}

/// Home page - Landing page with navigation
class HomePage extends StatelessWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final primaryColor = Theme.of(context).primaryColor;
    final secondaryColor = Theme.of(context).colorScheme.secondary;
    final isMobile = MediaQuery.of(context).size.width < 600;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Mushroom ID'),
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Get.toNamed('/settings'),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Padding(
            padding: EdgeInsets.all(isMobile ? 24.0 : 48.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: 40),

                // Logo/icon
                Container(
                  width: 100,
                  height: 100,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: primaryColor.withOpacity(0.1),
                  ),
                  child: Icon(
                    Icons.eco,
                    size: 50,
                    color: primaryColor,
                  ),
                ),
                const SizedBox(height: 24),

                // Title
                Text(
                  'Identify Mushrooms',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 12),

                // Subtitle
                Text(
                  'Use AI to identify mushroom species from photos',
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: Colors.grey[600],
                      ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 48),

                // Main action buttons
                SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: ElevatedButton.icon(
                    onPressed: () => Get.toNamed('/camera'),
                    icon: const Icon(Icons.camera_alt, size: 24),
                    label: const Text('Take Photo'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: primaryColor,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 12),

                SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: OutlinedButton.icon(
                    onPressed: () => Get.toNamed('/history'),
                    icon: const Icon(Icons.history, size: 24),
                    label: const Text('View History'),
                    style: OutlinedButton.styleFrom(
                      side: BorderSide(color: primaryColor, width: 2),
                      foregroundColor: primaryColor,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 48),

                // Recent identifications section (placeholder)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.grey[300]!),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Recent Identifications',
                        style: Theme.of(context).textTheme.labelLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Your previous identifications will appear here',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: Colors.grey[600],
                            ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 32),

                // Safety disclaimer
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.orange[50],
                    border: Border.all(color: Colors.orange[200]!),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            Icons.warning_amber_rounded,
                            color: Colors.orange[700],
                          ),
                          const SizedBox(width: 8),
                          Text(
                            'Safety Disclaimer',
                            style: Theme.of(context).textTheme.labelLarge?.copyWith(
                                  color: Colors.orange[900],
                                  fontWeight: FontWeight.bold,
                                ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'This app is for educational and informational purposes only. '
                        'Do NOT rely solely on this app for mushroom identification. '
                        'Always consult with expert mycologists or your local poison control '
                        'before consuming any wild mushroom.',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: Colors.orange[900],
                            ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
