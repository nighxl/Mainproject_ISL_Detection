import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'SignChartPage.dart'; // Import the new screen

class DetectionPage extends StatefulWidget {
  @override
  _DetectionPageState createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  CameraController? _cameraController;
  bool isDetecting = false;
  bool isFlashOn = false;
  String detectedSentence = "";
  final FlutterTts _flutterTts = FlutterTts();
  String serverUrl = "http://172.20.10.2:5001/predict";

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;
      _cameraController = CameraController(
        cameras[0],
        ResolutionPreset.high,
        enableAudio: false,
      );
      await _cameraController!.initialize();
      setState(() {});
    } catch (e) {
      print("❌ Camera error: $e");
    }
  }

  void _toggleFlash() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized)
      return;
    await _cameraController!.setFlashMode(
      isFlashOn ? FlashMode.off : FlashMode.torch,
    );
    setState(() {
      isFlashOn = !isFlashOn;
    });
  }

  void _startDetection() {
    if (!isDetecting && _cameraController?.value.isInitialized == true) {
      setState(() {
        isDetecting = true;
      });
      _continuousDetection();
    }
  }

  void _stopDetection() {
    setState(() {
      isDetecting = false;
    });
  }

  Future<void> _continuousDetection() async {
    while (isDetecting) {
      await Future.delayed(Duration(seconds: 10)); // 15-sec delay
      await _detectSign();
    }
  }

  Future<void> _detectSign() async {
    if (_cameraController == null ||
        !_cameraController!.value.isInitialized ||
        !isDetecting)
      return;
    try {
      final XFile image = await _cameraController!.takePicture();
      File processedImage = File(image.path);
      String prediction = await sendImageToServer(processedImage.path);
      if (prediction.isNotEmpty) {
        setState(() {
          detectedSentence += prediction;
        });
      }
    } catch (e) {
      print("❌ Detection error: $e");
    }
  }

  Future<String> sendImageToServer(String imagePath) async {
    try {
      var request = http.MultipartRequest("POST", Uri.parse(serverUrl));
      request.files.add(await http.MultipartFile.fromPath("image", imagePath));
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();
      var jsonResponse = json.decode(responseBody);
      return jsonResponse["prediction"] ?? "No sign detected";
    } catch (e) {
      print("❌ Server error: $e");
      return "Error";
    }
  }

  void _uploadImage() async {
    try {
      final pickedFile = await ImagePicker().pickImage(
        source: ImageSource.gallery,
      );
      if (pickedFile == null) return;
      File imageFile = File(pickedFile.path);
      String prediction = await sendImageToServer(imageFile.path);
      setState(() {
        detectedSentence += prediction;
      });
      _flutterTts.speak(detectedSentence);
    } catch (e) {
      print("❌ Upload error: $e");
    }
  }

  void _addSpace() {
    setState(() {
      detectedSentence += " ";
    });
  }

  void _clearLastWord() {
    setState(() {
      List<String> words = detectedSentence.trim().split(" ");
      if (words.isNotEmpty) {
        words.removeLast();
        detectedSentence = words.join(" ");
      }
    });
  }

  @override
  void dispose() {
    isDetecting = false;
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF1B9A8F),
      body: Column(
        children: [
          // Rounded Camera Preview
          Expanded(
            flex: 8,
            child: ClipRRect(
              borderRadius: BorderRadius.vertical(bottom: Radius.circular(30)),
              child:
                  _cameraController == null ||
                          !_cameraController!.value.isInitialized
                      ? Center(child: CircularProgressIndicator())
                      : CameraPreview(_cameraController!),
            ),
          ),
          // Sentence Display Box
          Container(
            width: double.infinity,
            margin: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
            padding: EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(15),
              boxShadow: [
                BoxShadow(
                  color: Colors.black38,
                  blurRadius: 6,
                  spreadRadius: 2,
                ),
              ],
            ),
            child: Text(
              detectedSentence,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          // Button Grid Layout
          Expanded(
            flex: 3,
            child: Padding(
              padding: EdgeInsets.all(10),
              child: GridView(
                shrinkWrap: true,
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3, // 3 buttons per row
                  crossAxisSpacing: 10,
                  mainAxisSpacing: 10,
                  childAspectRatio: 2, // Adjusted button aspect ratio
                ),
                children: [
                  _buildButton("Start", Colors.redAccent, _startDetection),
                  _buildButton("Stop", Colors.redAccent, _stopDetection),
                  _buildButton("Clear", Colors.redAccent, _clearLastWord),
                  _buildButton("Space", Colors.redAccent, _addSpace),
                  _buildButton("Upload", Colors.redAccent, _uploadImage),
                  _buildButton("Speak", Colors.redAccent, () {
                    if (!isDetecting) {
                      _flutterTts.speak(detectedSentence);
                    }
                  }),
                  _buildButton(
                    isFlashOn ? "Flash Off" : "Flash On",
                    Colors.redAccent,
                    _toggleFlash,
                  ),
                  _buildButton("Signs", Colors.redAccent, () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SignChartPage()),
                    );
                  }),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildButton(String label, Color color, VoidCallback onPressed) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      ),
      onPressed: onPressed,
      child: Text(label, style: TextStyle(color: Colors.white, fontSize: 16)),
    );
  }
}
