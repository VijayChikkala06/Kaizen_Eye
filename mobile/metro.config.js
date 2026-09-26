// Learn more https://docs.expo.dev/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Bundle the LiteRT backbone (.tflite) as an asset so react-native-fast-tflite can load it.
config.resolver.assetExts.push('tflite');

module.exports = config;
