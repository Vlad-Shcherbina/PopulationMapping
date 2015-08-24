const vector<vector<double>> prediction = { {0.0006278256666666667, 0.0008817755, 0.0010029, 0.0010942833333333333, 0.0011732374999999999, 0.0013295975, 0.0014093033333333331, 0.00145521, 0.0015275125000000001, 0.0015865733333333334, 0.00168115, 0.0017343475, 0.0018723425, 0.00194689, 0.00206275, 0.0021677375, 0.0022189400000000004, 0.00228292, 0.002542665, 0.0027699300000000003, 0.0028944649999999997, 0.0030472825, 0.00316804, 0.0032448925, 0.0034669175, 0.0038576025, 0.00450176, 0.0054370024999999995, 0.0058560975, 0.006075093333333333, 0.00641556, 0.0074376975, 0.00835488, 0.00983822, 0.011917674999999999, 0.053018499999999996}, {0.0020710345238095238, 0.002825492142857143, 0.003193641428571429, 0.003587937619047619, 0.003948808837209303, 0.00430339, 0.004631979523809524, 0.004947411190476191, 0.005223583953488372, 0.005535320714285715, 0.005885830714285715, 0.006187825714285714, 0.006554660476190476, 0.006951017209302326, 0.007388219523809523, 0.007937584047619049, 0.008445151666666666, 0.009081630930232558, 0.009609104523809524, 0.00997505119047619, 0.010571564285714286, 0.011290595238095238, 0.012067076744186046, 0.013132707142857145, 0.014420711904761904, 0.01563639285714286, 0.017430062790697674, 0.01966539761904762, 0.02273555, 0.027651421428571427, 0.034726457142857145, 0.04345056279069767, 0.055888073809523806, 0.07382941666666668, 0.10175348095238095, 0.20669662790697674}, {0.005479111111111112, 0.007376492222222222, 0.008669300181818182, 0.00980019537037037, 0.010775894545454546, 0.011665653703703704, 0.012578781818181818, 0.013350705555555556, 0.014126998181818182, 0.014970935185185186, 0.015823916666666667, 0.016864038181818183, 0.017977542592592592, 0.019182143636363634, 0.020167875925925925, 0.02135892181818182, 0.022512988888888887, 0.02374667090909091, 0.025319159259259257, 0.02722614259259259, 0.02928705090909091, 0.031495653703703705, 0.034084619999999996, 0.03815040925925926, 0.04322926363636364, 0.04798305185185185, 0.05431950181818182, 0.06016440185185185, 0.0685424574074074, 0.07975140181818181, 0.09389105185185186, 0.11292607272727273, 0.13352562962962963, 0.1621342727272727, 0.20754962962962964, 0.3778769090909091}, {0.009754203898305084, 0.01352125, 0.016170515, 0.018041538983050845, 0.020051866666666664, 0.02218189833333333, 0.024007979661016948, 0.026357056666666667, 0.028180185, 0.030046086440677966, 0.032314898333333335, 0.034619891666666666, 0.03670219322033898, 0.03905198333333334, 0.041810495, 0.04481239491525424, 0.048044841666666664, 0.0512439, 0.05579224915254237, 0.05991493833333333, 0.06510279000000001, 0.07054657118644068, 0.07692567833333334, 0.08521861333333333, 0.09314022033898305, 0.10278156833333334, 0.11324288333333334, 0.12757713559322034, 0.1403517833333333, 0.15444385, 0.1732004406779661, 0.19656433333333334, 0.22069688333333334, 0.25755815254237285, 0.33911569999999996, 0.5447059500000001}, {0.015581301739130434, 0.022223522857142856, 0.027714367142857143, 0.03153761, 0.03489553043478261, 0.03815201571428572, 0.04136625714285714, 0.04478740857142857, 0.04775305428571428, 0.05105448695652174, 0.05474636857142857, 0.058395835714285714, 0.06199740142857142, 0.06553183623188405, 0.06964307285714286, 0.07552456285714286, 0.08072963571428571, 0.08647649571428571, 0.09270043188405798, 0.09944682714285714, 0.10627125714285715, 0.1171396, 0.12800144927536233, 0.1389159, 0.1514518, 0.16446899999999998, 0.17847844285714287, 0.19457592753623187, 0.2089602142857143, 0.22700154285714286, 0.2463084142857143, 0.2693843188405797, 0.3004929142857143, 0.3468760857142857, 0.4255669, 0.5793685285714285}, {0.025065399642857145, 0.039082794117647056, 0.04737122117647059, 0.054114804761904764, 0.05993923176470588, 0.06492361176470587, 0.06964756785714285, 0.07422567529411765, 0.07885233176470588, 0.08344277142857143, 0.0885602305882353, 0.09460794470588235, 0.10022521176470588, 0.10688888095238096, 0.11397215294117646, 0.12156929411764705, 0.12964460714285714, 0.13774734117647058, 0.14671119999999999, 0.15625514285714287, 0.1659976705882353, 0.17693543529411765, 0.1895248095238095, 0.20375315294117646, 0.21691543529411766, 0.2309717411764706, 0.24879386904761905, 0.2660565764705882, 0.2845963411764706, 0.30165125000000004, 0.3213957411764706, 0.34777621176470586, 0.38127013095238094, 0.43067165882352937, 0.49957234117647054, 0.6328361411764706}, {0.03666683904761905, 0.056826520000000005, 0.06820753619047619, 0.07708148018867925, 0.08408651333333333, 0.09092035523809525, 0.09748824761904762, 0.1047749245283019, 0.11202092380952382, 0.11852690476190476, 0.12554228571428572, 0.1331786037735849, 0.14201470476190475, 0.15035840952380952, 0.15861141904761902, 0.16835258490566038, 0.17877761904761905, 0.18880581904761906, 0.19844226666666667, 0.2106165283018868, 0.22202245714285715, 0.23406605714285714, 0.2463114857142857, 0.260282641509434, 0.2756465238095238, 0.29138161904761906, 0.30822261904761905, 0.32423372641509435, 0.34410503809523807, 0.3688200380952381, 0.39531983809523813, 0.4212821320754717, 0.4564456952380952, 0.5005900571428572, 0.5584030285714285, 0.6863957075471698}, {0.05704255, 0.08491122430555556, 0.09940027222222222, 0.10999622222222222, 0.11948849305555556, 0.1287410138888889, 0.13732784027777778, 0.14631324827586206, 0.1553153263888889, 0.1648101111111111, 0.17515860416666668, 0.18439375694444443, 0.1937984027777778, 0.20365905555555555, 0.2149007172413793, 0.22589128472222222, 0.2360707291666667, 0.24782459027777776, 0.2601386319444445, 0.27252695833333335, 0.28497393055555553, 0.2991854344827586, 0.31256011111111115, 0.3274050277777778, 0.3423873125, 0.3583282361111111, 0.3746167083333333, 0.39307410416666666, 0.4117740206896552, 0.43214578472222226, 0.4581889236111111, 0.4900269305555556, 0.5253504999999999, 0.5671539930555556, 0.6144352222222222, 0.7038660689655173}, {0.08351904876847291, 0.11875911330049262, 0.1354088078817734, 0.14767922167487685, 0.15912459113300492, 0.17071563725490194, 0.18134414285714284, 0.19146003448275864, 0.20159757142857143, 0.2122886551724138, 0.22291052709359604, 0.2337334068627451, 0.24507549261083741, 0.25654460591133005, 0.2683608571428572, 0.2797386650246305, 0.2918809458128079, 0.3048806225490196, 0.3177044778325123, 0.3298549310344827, 0.3422024285714286, 0.35510193103448273, 0.3691920689655172, 0.3838140735294117, 0.39881116748768475, 0.41497304926108375, 0.43147304433497535, 0.44934334482758626, 0.46882120197044336, 0.4917032156862745, 0.5188883300492612, 0.548840487684729, 0.5797893891625616, 0.615984433497537, 0.6568521428571428, 0.7337930637254901}, {0.11393190508474577, 0.15547361016949152, 0.17445165423728812, 0.18907596621621622, 0.20293512203389832, 0.2162760406779661, 0.22960138983050848, 0.24280863851351353, 0.2556518101694915, 0.26757442033898304, 0.2793509391891892, 0.29185448135593217, 0.30408415254237287, 0.3151671389830508, 0.3274563918918919, 0.3393614745762712, 0.3516967084745763, 0.36386411486486486, 0.37630298983050847, 0.3893384271186441, 0.40343517627118647, 0.4178339831081081, 0.43277430847457626, 0.44718870847457626, 0.4622934101694915, 0.4769511824324324, 0.4934870983050848, 0.5139134508474577, 0.533234929054054, 0.5548043796610169, 0.5790161389830509, 0.6055382101694915, 0.632779304054054, 0.6637910677966101, 0.6977911898305085, 0.7596098378378379}, {0.15328792791762014, 0.20031436155606405, 0.2211077214611872, 0.23868680549199084, 0.2547919041095891, 0.2705836910755149, 0.28606987214611873, 0.3014576384439359, 0.31677792922374426, 0.3314976155606408, 0.34580816018306637, 0.35894886301369866, 0.37245347597254, 0.3852138378995434, 0.39824598398169336, 0.410888904109589, 0.4236421624713959, 0.4361291232876712, 0.4488335972540046, 0.46281257894736844, 0.475864196347032, 0.4891644370709382, 0.5031375114155251, 0.5170917070938215, 0.531179698630137, 0.5460427254004576, 0.5618756347031963, 0.577976, 0.596218590389016, 0.6134874086757991, 0.6332895034324943, 0.6545664611872146, 0.6770459931350115, 0.7029409315068493, 0.7303800457665904, 0.7856511621004566}, {0.19855011812778606, 0.2443258736998514, 0.2683341765578635, 0.29060918424962856, 0.3103535934718101, 0.3289266047548291, 0.3468572789317507, 0.36392476523031203, 0.3803596721068249, 0.39493417087667165, 0.4089610460624072, 0.4217913397626113, 0.43510774888558695, 0.4472599495548962, 0.45987656315007436, 0.4718738234421365, 0.48477516344725113, 0.4982424985163205, 0.5117325616641901, 0.5247273759286776, 0.5367903100890208, 0.5493393729569094, 0.5613760860534125, 0.5735795898959881, 0.5870429065281899, 0.5993961010401189, 0.6126414925816024, 0.6275258365527489, 0.6429363789004457, 0.6588234436201781, 0.6758715111441308, 0.6940184881305638, 0.7128755913818722, 0.732826206231454, 0.7564625393759287, 0.8016822329376855}, {0.21360096770833334, 0.2712598541666667, 0.3004182447916667, 0.32516279427083333, 0.34979383072916664, 0.3708838385416667, 0.391313453125, 0.41083252864583336, 0.4301538671875, 0.4472486848958333, 0.46236763541666664, 0.47633811458333336, 0.49058990625000004, 0.5037719453125, 0.5169160520833334, 0.529784015625, 0.5440600911458333, 0.5574560207792207, 0.5704725260416666, 0.5820071484375, 0.5949329375, 0.6071061015625, 0.6195345546875001, 0.63216178125, 0.6456011119791667, 0.6592751458333334, 0.6723346197916666, 0.6864970859375, 0.701022546875, 0.7161668229166667, 0.7315557968749999, 0.7471497369791668, 0.7631860260416666, 0.7805599166666667, 0.8014788854166667, 0.8479969168831168}, {0.23415431822222224, 0.3044340311111111, 0.33920206637168143, 0.3690656755555556, 0.3938268672566372, 0.4210602577777778, 0.44618848, 0.4688353141592921, 0.48850956, 0.5061658849557522, 0.5220145822222222, 0.5380642920353982, 0.555457408888889, 0.5698320977777778, 0.585078424778761, 0.6001413200000001, 0.6133854070796461, 0.6258673511111111, 0.6385811022222222, 0.6505937300884956, 0.6621326622222222, 0.6745413672566372, 0.6866198, 0.6995039115044248, 0.7117835466666667, 0.7247306177777778, 0.7367408849557523, 0.74837928, 0.7598964380530974, 0.7728591777777778, 0.7842022488888889, 0.7971699955752212, 0.8110810533333334, 0.8262151548672566, 0.8457561333333333, 0.8883057743362832}, {0.25121954507042255, 0.3331268951048951, 0.3792784405594406, 0.4133093216783217, 0.44443235211267607, 0.4736079440559441, 0.4983494825174825, 0.520664, 0.5414092605633803, 0.5598028951048951, 0.5776715314685315, 0.5942463006993007, 0.609255028169014, 0.625138986013986, 0.6398172657342658, 0.6546610209790209, 0.6668316056338028, 0.6805167202797203, 0.6929071748251748, 0.7031321468531468, 0.7144685211267606, 0.7262757622377622, 0.739073993006993, 0.7512276433566433, 0.7622165563380282, 0.7737525734265734, 0.7856030909090909, 0.7970065664335665, 0.8083152323943662, 0.8198939230769231, 0.8307673706293707, 0.842157951048951, 0.8536602605633803, 0.8660735524475525, 0.8831352867132868, 0.9170314405594405}, {0.2770241774193548, 0.3680801914893617, 0.4207956170212766, 0.45993490425531913, 0.49360460638297876, 0.5280743723404255, 0.5570787234042553, 0.5801046666666667, 0.6012468723404255, 0.6233119574468086, 0.6423499680851064, 0.6585146595744681, 0.6742269787234042, 0.6916681276595744, 0.7075302150537633, 0.7200210531914893, 0.7319875957446808, 0.7440104787234043, 0.7541792234042554, 0.7654436276595745, 0.7767630425531915, 0.7864583010752687, 0.7960223829787234, 0.804931670212766, 0.8141817340425532, 0.8251741808510639, 0.8345962021276596, 0.8444998297872339, 0.8526483655913978, 0.860339074468085, 0.8697334680851064, 0.8786126808510638, 0.8880837765957446, 0.8999505, 0.9138295425531914, 0.942908}, {0.30871314285714285, 0.42248173015873014, 0.48223334920634925, 0.5365046349206349, 0.5724773809523809, 0.6038801587301587, 0.6345466031746032, 0.6609856825396826, 0.6824456349206349, 0.6996457301587301, 0.7152177619047619, 0.7263226031746032, 0.7396478888888889, 0.7518415238095238, 0.762992746031746, 0.7748331428571428, 0.7876920476190477, 0.799427890625, 0.8119703968253967, 0.8213194444444444, 0.8299258412698413, 0.8396549365079364, 0.8483329047619047, 0.8565167301587302, 0.8658152222222223, 0.8730715396825397, 0.8791149206349206, 0.8861508412698412, 0.8934901428571428, 0.8993563333333333, 0.9064911904761904, 0.9133728571428572, 0.9200447301587301, 0.929399746031746, 0.940326746031746, 0.960526234375}, {0.3177805777777778, 0.44127441304347825, 0.5226555652173913, 0.5826077391304348, 0.6326697608695653, 0.666455652173913, 0.6937025217391304, 0.7230399999999999, 0.7459385652173912, 0.7681611956521739, 0.7864520217391305, 0.8013144347826087, 0.8130273043478261, 0.8229463695652174, 0.8340569130434783, 0.8425640869565217, 0.8520733913043479, 0.8606368913043478, 0.870539652173913, 0.8777884782608696, 0.8849220869565216, 0.8917112826086956, 0.8986204782608695, 0.9042370869565218, 0.9090298260869565, 0.9136713043478261, 0.9179547391304348, 0.9223769347826086, 0.9262216086956522, 0.9299558695652175, 0.9344458260869565, 0.9384581304347827, 0.9434573478260869, 0.9495368913043478, 0.9584070652173914, 0.9732660434782608}, {0.38872754545454546, 0.5410834545454546, 0.6183400882352941, 0.6725772121212121, 0.7151488529411765, 0.7453471818181818, 0.7694191515151515, 0.791415705882353, 0.8080026666666666, 0.8244264705882353, 0.8403977575757575, 0.8526116764705883, 0.8607100606060606, 0.8705726363636364, 0.8780436764705882, 0.8844406363636363, 0.8927757941176471, 0.8988410606060606, 0.9060867575757576, 0.9120357352941176, 0.9162348484848485, 0.9201142058823529, 0.9248542424242424, 0.9310582058823529, 0.9366544848484848, 0.939816606060606, 0.943353705882353, 0.9469968484848486, 0.9494276470588235, 0.9524593939393939, 0.9561998787878788, 0.9600230000000001, 0.9630368181818182, 0.9673731176470589, 0.9739490000000001, 0.9825883823529412}, {0.39423245833333337, 0.59674012, 0.67677768, 0.73603776, 0.7779912, 0.80697564, 0.830065125, 0.84877948, 0.8639571600000001, 0.87982172, 0.89049992, 0.90011608, 0.9089764583333334, 0.91605, 0.9235579199999999, 0.92979912, 0.93486636, 0.9383501200000001, 0.9420902916666667, 0.94544556, 0.94826992, 0.9510573600000001, 0.953863, 0.9564952800000001, 0.959109375, 0.96151692, 0.96333428, 0.96597152, 0.96831848, 0.97100152, 0.9732227916666667, 0.97584648, 0.9784082000000001, 0.98096856, 0.98355024, 0.98809448}, {0.5459145555555556, 0.7570143157894736, 0.8300430555555556, 0.8581235263157895, 0.8815590555555555, 0.896037947368421, 0.9119492222222222, 0.9253255789473683, 0.9323551111111112, 0.9402107894736842, 0.9446614999999999, 0.9489494210526317, 0.9535532777777777, 0.956534052631579, 0.9597058888888887, 0.9624666315789474, 0.9656752222222222, 0.9681257894736843, 0.9700904736842105, 0.971983, 0.9745004736842106, 0.9761557222222224, 0.9779892631578948, 0.9795395, 0.9806373684210526, 0.9815690555555556, 0.9824421052631579, 0.9835569444444445, 0.9846235789473683, 0.9858213333333333, 0.9866493684210527, 0.987921, 0.9891272105263159, 0.9905077222222223, 0.9917776842105263, 0.9938707368421053}, {0.7333992142857142, 0.8938184285714286, 0.9283452666666666, 0.9398653571428571, 0.9503466428571429, 0.958306, 0.9655132142857142, 0.9743101428571429, 0.9781454, 0.9801712857142857, 0.9819248571428572, 0.9836702666666667, 0.985228, 0.9862074, 0.9871567142857144, 0.9880726428571428, 0.9889130666666667, 0.9896205, 0.9902759999999999, 0.9909036666666666, 0.9913881428571428, 0.9917700714285714, 0.9921542666666666, 0.9927520000000001, 0.9932412, 0.9936838571428571, 0.993961642857143, 0.9942508666666666, 0.9946883571428572, 0.9950469285714286, 0.9953440666666666, 0.9956476428571428, 0.9960640714285713, 0.9966832, 0.9970632857142857, 0.9978497333333334}, {0.819304, 0.963627, 0.968086, 0.976351, 0.982323, 0.983403, 0.986673, 0.990148, 0.990672, 0.990954, 0.992672, 0.9941, 0.996397, 0.996411, 0.996432, 0.996494, 0.996593, 0.996653, 0.997001, 0.997438, 0.997544, 0.997662, 0.997795, 0.998017, 0.998101, 0.998129, 0.99814, 0.998151, 0.998201, 0.998309, 0.998626, 0.998633, 0.998703, 0.998713, 0.998843, 0.998892} };