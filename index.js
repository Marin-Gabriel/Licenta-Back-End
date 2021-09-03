const Express = require('express')
const multer = require('multer')
const bodyParser = require('body-parser')
const mongoose = require('mongoose')
const User = require('./models/user')
const ImageSet = require('./models/imageSet')
const {v4:uuidv4} = require('uuid');
const jwt = require('jsonwebtoken')
const app = Express()
require('dotenv').config();
var bcrypt = require('bcryptjs')

const dbURI='mongodb://localhost:27017/OCR'
mongoose.connect(dbURI,{useNewUrlParser:true, useUnifiedTopology:true})

mongoose.connection
.once('open',()=>console.log('Connected'))
.on('error',error =>{
  console.log(error)
})  
app.use(bodyParser.urlencoded({extended:false}))
app.use(bodyParser.json())

var allowCrossDomain = function(req, res, next) {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  next();
}

app.use(allowCrossDomain);

app.use(Express.json());

app.use(bodyParser.urlencoded({
  extended: true
}));

const Storage = multer.diskStorage({
  destination(req, file, callback) {
    callback(null, './images')
  },
  filename(req, file, callback) {
    callback(null, `${file.fieldname}_${Date.now()}_${file.originalname}`)
  },
})

const upload = multer({ storage: Storage })

app.get('/',validateToken, (req, res) => {
  res.status(200).send('You can post to /api/upload.')
})

const findMaxSetId = async function (UID) {
  try{
    return await ImageSet.find({UID:UID})
    .sort({setId:-1})
    .limit(1)
  }
  catch(err)
  {
    console.log(err)
  }
}

const findImageSets = async function (UID) {
  try{
    return await ImageSet.find({UID:UID},{setId:1,originalBase64Data:1,processedBase64Data:1})
    .sort({setId:1})
  }
  catch(err)
  {
    console.log(err)
  }
}

const findImageSet = async function(UID,SetId){
  try{
    return await ImageSet.find({UID:UID,setId:SetId},{setId:1})
  }
  catch(err)
  {
    console.log(err)
  }
}

const deleteImageSet = async function (UID,setId){
  try{
    return await ImageSet.deleteOne({UID:UID,setId:setId})
  }
  catch(err)
  {
    console.log(err)
  }
}

app.get('/api/getImages',validateToken,(req,res)=>{
  let UID = req.UID
  findImageSets(UID).then(function(result){
    if(result.length>0){
      var arr = []
      result.forEach(element => arr.push({image:'data:image/png;base64,'+element.originalBase64Data,processedImage:'data:image/png;base64,'+element.processedBase64Data,key:element.setId.toString()}));
      res.status(200).send(arr)
    }
    else
    {
      res.status(404)
    }
  })
})

app.delete('/api/delete',validateToken, (req, res) => {

  let UID=req.UID
  let SetId=req.query.SetId

  findImageSet(UID,SetId).then(function(result){
    if(result.length>0)
    {
      deleteImageSet(UID,SetId).then(function(resultDelete){
        if(resultDelete.deletedCount==1){
          res.status(200).send(JSON.stringify({deleted:true}))
        }
        else{
          res.status(500)
        }
      })
      .catch((err) => {
        res.status(500)
      });
    }
    else
    {
      res.status(404)
    }
  })
})

app.delete('/api/discard',validateToken, (req, res) => {
  let UID = req.UID
  findMaxSetId(UID).then(function(resultFind){
      if(resultFind.length>0){
        deleteImageSet(UID,resultFind[0].setId).then(function(resultDelete){
          if(resultDelete.deletedCount==1){
            res.status(200).send(JSON.stringify({deleted:true}))
          }
          else{
            res.status(200).send(JSON.stringify({deleted:false}))
          }
        })
        .catch((err) => {
          res.status(500)
        });
      }
      else
      {
        res.status(200).send(JSON.stringify({deleted:false}))
      }
    })
  })

function validateToken(req,res,next){

  const authorizationHeader = req.headers['authorization']
  const token = authorizationHeader && authorizationHeader.split(' ')[1]

  if(token == null ){
    return res.sendStatus(401)
  }

  jwt.verify(token,process.env.TOKEN_SECRET,(err,uid) => {
    if(err) {
      return res.sendStatus(403)
    }  
    req.UID = uid.UID
    next()
  })
}

app.post('/login', (req, res) => {

  User.find({username: req.body.username},(err,user) =>{
    if(err) {
      res.status(500)
    }
    if(user.length){
      bcrypt.compare(req.body.password,user[0].password,function(err,result){
        if(err){
          res.status(500)
        }
        else if(result){
          const UID = { UID: user[0].UID}
          const accessToken = jwt.sign(UID,process.env.TOKEN_SECRET)
          res.status(200).send(JSON.stringify({accessToken : accessToken,ok:true}))
        }
        else{
          res.status(404).send(JSON.stringify({ok:false}))
        }
      })
    }
    else{
    res.status(404).send(JSON.stringify({ok:false}))
    }
  })
})

app.post('/register', (req, res) => {

  User.find({username: req.body.username},(err,user) =>{
    if(err){
      res.status(500)
    }
    if(user.length){
      res.status(409).send(JSON.stringify({ok:false}))
    }
    else{
      bcrypt.genSalt(10,function(err,salt){
        bcrypt.hash(req.body.password,salt,function(err,hash){
          if(err){
            res.status(500)
          }
          else{
          const user = new User({
            username:req.body.username,
            password:hash,
            UID:uuidv4()
          });
          user.save()
            .then(() => {
              res.status(201).send(JSON.stringify({ok:true}))
            })
            .catch((err) => {
              res.status(500)
            });
          }
        })
      })
    }
  })
})

app.post('/api/upload',validateToken, upload.array('photo', 3), (req, res) => {

  const start = Date.now()
  var tesseractResult='';
  var originalBase64='';

  const spawn = require('child_process').spawn
  const imageToBase64=spawn('python',['imageToBase64.py',req.files[0].filename])
  imageToBase64.stdout.on('data',(chunk)=>{
    originalBase64+=chunk;
  });

  const python=spawn('python',['Tesseract2.7.py',req.files[0].filename,req.body.language])
  python.stdout.on('data',(chunk)=>{
    tesseractResult+=chunk;
  });
  python.on('exit', () => {

    findMaxSetId(req.UID).then(function(result){
      if(result.length>0){
        const imageSet = new ImageSet({
          UID:req.UID,
          setId:result[0].setId+1,
          originalBase64Data:originalBase64,
          processedBase64Data:tesseractResult
        });

        imageSet.save()
        .catch((err) => {console.log(err)});

        res.status(200).json({
          message: tesseractResult,
          newImageSet:{image:'data:image/png;base64,'+originalBase64,processedImage:'data:image/png;base64,'+tesseractResult,key:(result[0].setId+1).toString()}
        })
      }
      else
      {
        const imageSet = new ImageSet({
          UID:req.UID,
          setId:0,
          originalBase64Data:originalBase64,
          processedBase64Data:tesseractResult
        });

        imageSet.save()
        .catch((err) => {console.log(err)});

        res.status(200).json({
          message: tesseractResult,
          newImageSet:{image:'data:image/png;base64,'+originalBase64,processedImage:'data:image/png;base64,'+tesseractResult,key:'0'}
        })
      }
    })
    const stop = Date.now()
    console.log(`Timp de procesare = ${(stop - start)/1000} secunde`);
  })
})

app.get('/get',(rez,res)=>{
  res.end('Hello');
})

app.listen(10000,'192.168.0.102', () => {
  console.log('App running on http://192.168.0.102:10000')
})