import 'package:flutter/material.dart';

class SignChartPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Sign Chart"),
        backgroundColor: Color(0xFF1B9A8F), // Matches the app's theme
      ),
      body: Center(
        child: InteractiveViewer(
          panEnabled: true, // Allows dragging
          minScale: 0.5, // Minimum zoom level
          maxScale: 3.0, // Maximum zoom level
          child: Image.asset(
            "asset/signs.png",
          ), // Ensure "signs.png" is in the assets folder
        ),
      ),
    );
  }
}
