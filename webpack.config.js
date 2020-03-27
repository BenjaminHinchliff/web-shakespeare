const path = require('path');

module.exports = {
  entry: './src/index.js',
  output: {
    path: path.resolve(__dirname),
    filename: 'index.bundle.js',
  },
  devServer: {
    contentBase: path.resolve(__dirname),
    watchContentBase: true,
    compress: true,
    port: 9000,
  },
};
