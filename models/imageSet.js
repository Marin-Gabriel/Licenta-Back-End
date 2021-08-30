const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const imageSetSchema = new Schema({
    UID:{
        type:String,
        required:true
    },
    setId:{
        type:Number,
        required:true
    },
    originalBase64Data:{
        type:String,
        required:true
    },
    processedBase64Data:{
        type:String,
        required:true
    }
}, {timestamps:true});

const ImageSet = mongoose.model('ImageSet', imageSetSchema);

module.exports = ImageSet;