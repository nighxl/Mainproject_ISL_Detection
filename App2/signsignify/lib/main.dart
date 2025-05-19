import 'package:flutter/material.dart';
import 'Landing.dart'; // Import the Landing screen

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SignSignify',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home:
          const MainScreen(), // Make sure Landing.dart has a MainScreen widget
    );
  }
}
