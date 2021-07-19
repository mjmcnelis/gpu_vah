
#ifndef TRANSPORTANISONONCONFORMAL_CUH_
#define TRANSPORTANISONONCONFORMAL_CUH_

#include "Precision.h"
#include "Macros.h"
#include "DynamicalVariables.h"
#include "Parameters.h"

// cuda: set class to device (do I need to worry about globals?)

const precision g = 51.4103536012791;			// quasiparticle degeneracy factor (Eq. 46)
												// g = (2.(Nc.Nc - 1) + 4.Nc.Nf.7/8) . pi^4 / 90

const int pbar_pts = 16;

const precision pbar_root_a0[pbar_pts] =   {0.08764941047892784036,
											0.46269632891508083188,
											1.1410577748312268569,
											2.1292836450983806163,
											3.4370866338932066452,
											5.0780186145497679129,
											7.0703385350482341304,
											9.4383143363919387839,
											12.214223368866158737,
											15.441527368781617077,
											19.180156856753134855,
											23.515905693991908532,
											28.578729742882140368,
											34.583398702286625815,
											41.940452647688332635,
											51.701160339543318364};

const precision pbar_weight_a0[pbar_pts] = {0.20615171495780099433,
											0.33105785495088416599,
											0.2657957776442141526,
											0.13629693429637753998,
											0.047328928694125218978,
											0.011299900080339453231,
											0.0018490709435263108643,
											0.00020427191530827846013,
											0.000014844586873981298771,
											6.8283193308711995644e-7,
											1.8810248410796732139e-8,
											2.8623502429738816196e-10,
											2.1270790332241029674e-12,
											6.2979670025178677872e-15,
											5.0504737000355128204e-18,
											4.1614623703728551904e-22};
//:::::::::::::::::::::::::::::


const precision pbar_root_a2[pbar_pts] =   {0.37761350834474073436,
											1.0174919576025657044,
											1.947758020424243826,
											3.1769272448898686839,
											4.7162400697917956248,
											6.5805882657749125043,
											8.7894652706470741271,
											11.368323082833318905,
											14.350626727437044399,
											17.781095724841646049,
											21.721084796571308914,
											26.258138675111068012,
											31.524596004275818683,
											37.738921002528939074,
											45.318546110089842558,
											55.3325835388358122};

const precision pbar_weight_a2[pbar_pts] = {0.04860640946707870253,
											0.29334739019044023149,
											0.58321936338355098941,
											0.5818741485961734789,
											0.33818053747379021731,
											0.12210596394497989094,
											0.028114625800663730181,
											0.004143149192482256704,
											0.00038564853376743830033,
											0.000022015800563109071234,
											7.3423624381565175075e-7,
											1.3264604420480413669e-8,
											1.1526664829084294661e-10,
											3.9470691512460869719e-13,
											3.6379782563605336032e-16,
											3.4545761231361240027e-20};

//:::::::::::::::::::::::::::::

const precision pbar_root_a3[pbar_pts] =   {0.56744345899157414477,
											1.3329077327598933409,
											2.3814824800700584858,
											3.7238266420934269684,
											5.3721239521618709961,
											7.3419366282613523442,
											9.6533321372612290461,
											12.332301407018237936,
											15.412850065407715512,
											18.940275575860272298,
											22.976595715601869244,
											27.610181447426074872,
											32.974509203258495494,
											39.289823253364147934,
											46.976896276710312577,
											57.113514023753468805};

const precision pbar_weight_a3[pbar_pts] = {0.065098112100944931481,
											0.56527322423640238298,
											1.4898931385790650188,
											1.8612470448965348651,
											1.3004755484827235985,
											0.54781964685691509622,
											0.14384304320810570104,
											0.023749847242162275954,
											0.0024424446935061086306,
											0.00015234059955885076823,
											5.5012653050135610492e-6,
											1.0684262964359744438e-7,
											9.9252433589722221362e-10,
											3.6186334278007682335e-12,
											3.5437327807321532759e-15,
											3.5818035528777921894e-19};

//:::::::::::::::::::::::::::::

const precision pbar_root_a4[pbar_pts] =   {0.78233916408563591043,
											1.6700718367187400142,
											2.8329217194725773694,
											4.2844183681902189091,
											6.0378687109688947902,
											8.1095493341397402125,
											10.520101704850221753,
											13.296039912356318355,
											16.471888708696199626,
											20.093499973053039387,
											24.223529942806254945,
											28.951149580651938334,
											34.410943040802021628,
											40.824895271382048087,
											48.617054973472911275,
											58.873727758353239406};

const precision pbar_weight_a4[pbar_pts] = {0.12973970266305223231,
											1.4974866580833850765,
											4.9461261428120737432,
											7.4302670508678579631,
											6.0568766623187282913,
											2.9091613135775531995,
											0.85558465181983766561,
											0.15600959865281362952,
											0.017519135965089154176,
											0.0011822725730000712572,
											0.000045850631165934906113,
											9.506160798276387613e-7,
											9.3830975950340207876e-9,
											3.6228100077043250866e-11,
											3.7511919062870009016e-14,
											4.0167067539283596667e-18};

/*
const int pbar_pts = 20;

const precision pbar_root_a0[pbar_pts] =   {0.070539889691988753367,
											0.37212681800161144379,
											0.91658210248327356467,
											1.7073065310283438807,
											2.7491992553094321296,
											4.0489253138508869224,
											5.6151749708616165141,
											7.4590174536710633098,
											9.5943928695810967725,
											12.03880254696431631,
											14.814293442630739979,
											17.948895520519376017,
											21.478788240285010976,
											25.451702793186905504,
											29.932554631700612007,
											35.013434240479000006,
											40.833057056728571062,
											47.61999404734650214,
											55.810795750063898891,
											66.524416525615753819};

const precision pbar_weight_a0[pbar_pts] = {0.16874680185111386215,
											0.29125436200606828172,
											0.26668610286700128855,
											0.16600245326950684003,
											0.07482606466879237054,
											0.024964417309283221073,
											0.0062025508445722368474,
											0.001144962386476908242,
											0.00015574177302781197478,
											0.000015401440865224915689,
											1.0864863665179823515e-6,
											5.3301209095567147509e-8,
											1.7579811790505820036e-9,
											3.7255024025123208726e-11,
											4.7675292515781905245e-13,
											3.3728442433624384124e-15,
											1.155014339500398831e-17,
											1.5395221405823435535e-20,
											5.2864427255691578288e-24,
											1.6564566124990232959e-28};
//:::::::::::::::::::::::::::::

const precision pbar_root_a2[pbar_pts] =   {0.30713030449433458782,
											0.82671116019541575664,
											1.5801236748240223804,
											2.5719669210344059791,
											3.8080377553304242113,
											5.2957813385708740928,
											7.0445977004594117415,
											9.0662191178153444839,
											11.375220078771874714,
											13.989724926712798672,
											16.932411393991265963,
											20.231971584801627102,
											23.925310505380243591,
											28.060995738545914332,
											32.704966371571631847,
											37.950658303403634889,
											43.938717535583635735,
											50.900824090462422347,
											59.2794680217571272,
											70.209163476293590375};

const precision pbar_weight_a2[pbar_pts] = {0.028045547579310043326,
											0.19009986758945737902,
											0.4482827043797567495,
											0.56231744128449456462,
											0.43782023215688387406,
											0.22725641118202038892,
											0.081497353883286753088,
											0.020531585378405573759,
											0.0036506945731492668201,
											0.0004561659865731272363,
											0.000039568251820362288557,
											2.3345448848813134463e-6,
											9.0934663707078512713e-8,
											2.2420541414230366339e-9,
											3.2972741791668394063e-11,
											2.6541799641843548566e-13,
											1.0262078187515314366e-15,
											1.536212035356379976e-18,
											5.91331156548931674e-22,
											2.0884244786089446187e-26};

//:::::::::::::::::::::::::::::

const precision pbar_root_a3[pbar_pts] =   {0.46370782790373323452,
											1.0879365342528900453,
											1.9405712784389576973,
											3.027812109866941466,
											4.3560914219617323694,
											5.9332762369167235683,
											7.769111374415115925,
											9.8756380549276704953,
											12.267722251442431294,
											14.963774224283484363,
											17.986762666457521848,
											21.365687646522072614,
											25.137794886052646892,
											29.352048579944572913,
											34.074877594457856037,
											39.400366486161910754,
											45.470097552177351329,
											52.517264953775725021,
											60.987258759214974563,
											72.022199560825687571};

const precision pbar_weight_a3[pbar_pts] = {0.032175523548689894035,
											0.31985447258210776965,
											1.0169815413585005822,
											1.6217463112774385879,
											1.5389159723159192168,
											0.94350672791587640516,
											0.39018312307854083567,
											0.11124448373660780825,
											0.02205130231454353615,
											0.0030344693323113967509,
											0.00028697866656075817137,
											0.000018307672263664985235,
											7.6572769094505365666e-7,
											2.0155325814139498085e-8,
											3.1492890430670171295e-10,
											2.6829183017865242835e-12,
											1.0945241426193170192e-14,
											1.7254042031267574984e-17,
											6.9904129151010991476e-21,
											2.6057993361062917579e-25};

//:::::::::::::::::::::::::::::

const precision pbar_root_a4[pbar_pts] =   {0.64213248527444626441,
											1.3689504083008093635,
											2.3180247754073767073,
											3.4977897206592333066,
											4.9154830087829395577,
											6.5794461369656446972,
											8.4997818998698949489,
											10.688838501466587627,
											13.161765315346900918,
											15.937248013481380917,
											19.038534902999370999,
											22.4949224614902706,
											26.343985512700829205,
											30.635073035223966157,
											35.435091177975865285,
											40.838759097159017562,
											46.988577976531215075,
											54.11923130669616117,
											62.678965623745964391,
											73.817398639922125248};

const precision pbar_weight_a4[pbar_pts] = {0.05554010868754105662,
											0.74697508369010272266,
											3.0227571425527306478,
											5.8775731625484642643,
											6.5890531745956574519,
											4.6587004360184342114,
											2.1801690135562691812,
											0.6928494399606351988,
											0.15121686166297836625,
											0.022681106648827211387,
											0.0023184437960984893258,
											0.00015874112968710378816,
											7.0838945896726234689e-6,
											1.9795819425709795791e-7,
											3.270306675237720387e-9,
											2.9357402642851579078e-11,
											1.258785646550254489e-13,
											2.0821628874261226485e-16,
											8.8491378845511300862e-20,
											3.4700598479868877273e-24};
*/


/*
// put the arrays here for now (can I get away with 16 points?)
const int pbar_pts = 32;

//:::::::::::::::::::::::::::::

const precision pbar_root_a0[pbar_pts] = {0.044489365833267,0.234526109519619,0.576884629301886,1.07244875381782,1.72240877644465,2.52833670642579,3.49221327302199,4.61645676974977,5.90395850417424,7.35812673318624,8.9829409242126,10.78301863254,12.7636979867427,14.9311397555226,17.2924543367153,19.8558609403361,22.6308890131968,25.6286360224592,28.8621018163235,32.3466291539647,36.100494805752,40.1457197715394,44.5092079957549,49.2243949873086,54.3337213333969,59.892509162134,65.975377287935,72.6876280906627,80.1874469779135,88.7353404178924,98.829542868284,111.751398097938};

const precision pbar_weight_a0[pbar_pts] = {0.109218341952385,0.210443107938813,0.235213229669848,0.195903335972881,0.129983786286072,0.0705786238657174,0.0317609125091751,0.0119182148348386,0.00373881629461152,0.000980803306614955,0.000214864918801364,3.92034196798795e-05,5.93454161286863e-06,7.41640457866755e-07,7.60456787912078e-08,6.35060222662581e-09,4.28138297104093e-10,2.30589949189134e-11,9.79937928872709e-13,3.23780165772927e-14,8.17182344342072e-16,1.54213383339382e-17,2.11979229016362e-19,2.05442967378805e-21,1.3469825866374e-23,5.66129413039736e-26,1.41856054546304e-28,1.91337549445422e-31,1.19224876009822e-34,2.67151121924014e-38,1.33861694210626e-42,4.51053619389897e-48};

//:::::::::::::::::::::::::::::

const precision pbar_root_a2[pbar_pts] = {0.196943922146667,0.529487866050161,1.01026981913845,1.640616191672,2.42200673335506,3.35625823737525,4.44557319147359,5.69257570606939,7.10035048878373,8.67248915845674,10.413146435518,12.3271087558129,14.4198784243951,16.6977773650005,19.1680758788069,21.839153763432,24.7207039368187,27.823992811746,31.1621978174102,34.7508519173206,38.6084399084037,42.7572156420076,47.2243504952188,52.0435960848824,57.257778984273,62.9227106235616,69.1136582681551,75.9368320953467,83.5517824825995,92.221284870548,102.447989923982,115.52490220024};

const precision pbar_weight_a2[pbar_pts] = {0.00825033790777967,0.0671033262747106,0.206386098255352,0.368179392999486,0.446389764546666,0.397211321904435,0.270703020914857,0.144937243765141,0.0619302157291065,0.0213227539141068,0.00594841159169929,0.00134795257769464,0.000248166548996264,3.7053223540482e-05,4.47057760459712e-06,4.33555258401213e-07,3.35571417159735e-08,2.05432200435071e-09,9.83646900727572e-11,3.63364388210833e-12,1.01834576904109e-13,2.12110313498633e-15,3.20100105319804e-17,3.39007439648141e-19,2.41904571899768e-21,1.10270714408855e-23,2.98827103874582e-26,4.34972188455989e-29,2.92108431650778e-32,7.0533942409897e-36,3.81617106981223e-40,1.39864930768275e-45};

//:::::::::::::::::::::::::::::

const precision pbar_root_a3[pbar_pts] = {0.299618729049241,0.701981065353977,1.24974569814569,1.94514382443706,2.78994869155499,3.78614305529879,4.93605293430173,6.24240692441721,7.70838362900271,9.3376617765878,11.1344784990019,13.1036993239838,15.2509033992287,17.5824881879873,20.1057991514724,22.8292918399089,25.7627366009885,28.9174802233293,32.3067850039226,35.9462752181738,39.8545359681502,44.0539338428991,48.5717701879893,53.4419507581545,58.7074908793654,64.4244418290391,70.6683893377061,77.5459900633602,85.2174664086695,93.9467116599065,104.238552969691,117.391742318923};

const precision pbar_weight_a3[pbar_pts] = {0.00660146448073508,0.0813584931042281,0.347537436309438,0.809963198105261,1.22739584119905,1.32050782861975,1.06049919505728,0.655616488144915,0.318173017008472,0.122743109012855,0.0379333897858022,0.00943187028689987,0.00188978713293874,0.000304914974586437,3.95130877631855e-05,4.09377958251348e-06,3.36921618654073e-07,2.1841295448875e-08,1.10337736506627e-09,4.28638379146177e-11,1.25966453444067e-12,2.74423030367617e-14,4.32175197361363e-16,4.76686817705967e-18,3.53643350342934e-20,1.67355018349782e-22,4.70254099995936e-25,7.09116556196869e-28,4.93082516196282e-31,1.23284946609868e-34,6.91389702736573e-39,2.63586492716958e-44};

//:::::::::::::::::::::::::::::

const precision pbar_root_a4[pbar_pts] = {0.4179092349285205164,0.88953757977412956343,1.5031343535821892418,2.2622772467922400713,3.1691609081884946231,4.2259814223739388132,5.4352075859025698526,6.7996801473429795616,8.3226739971421950501,10.00795481066206095,11.859840697138091813,13.883274215094546335,16.08390900396634473,18.468215606636866113,21.043612172303404879,23.818627569275657412,26.803107200065090088,30.008475918731580888,33.448078637568861185,37.137628733701203404,41.095809422480242898,45.345097823469503332,49.912922981558044551,54.833342392061308785,60.149557510492311786,65.917856460530624905,72.214139807179640278,79.145505977910217234,86.872842716877578922,95.661150475367234798,	106.0174125440344689,119.24607484686785852};

const precision pbar_weight_a4[pbar_pts] = {0.0081035021776870674224,0.13934980269565479628,0.77892765820920495419,2.270688477817657329,4.162019786788386281,5.2778864167004716975,4.8948590924479342971,3.437172809206616944,1.8690454619642390358,0.79876388700087097392,0.27085216482965314289,0.073287879591448861235,0.015867148817090025692,0.002749473720138891225,0.00038059786010185430023,0.000041923441122147192945,3.6530960788398492826e-6,2.4980951271919747913e-7,1.3268752725228344442e-8,5.403883159066006477e-10,1.6605449308744093245e-11,3.7739278121221075136e-13,6.1875744797835908925e-15,7.0924760328904819544e-17,5.4595039769778605091e-19,2.6771369913592442615e-21,7.786426605077152367e-24,1.2143682326490162623e-26,8.729513743508775062e-30,2.2566202260183591845e-33,1.3097904910237646072e-37,5.1860151170004670833e-43};
*/

__device__
class aniso_transport_coefficients_nonconformal
{
	// nonconformal vahydro transport coefficients
	private:

	public:
		precision zeta_LL;			// pl coefficients
		precision zeta_TL;
		precision lambda_WuL;
		precision lambda_WTL;
		precision lambda_piL;

		precision zeta_LT;			// pt coefficients
		precision zeta_TT;
		precision lambda_WuT;
		precision lambda_WTT;
		precision lambda_piT;

	#ifdef WTZMU
		precision eta_uW;			// WTz coefficients
		precision eta_TW;
		precision tau_zW;
		precision delta_WW;
		precision lambda_WuW;
		precision lambda_WTW;
		precision lambda_piuW;
		precision lambda_piTW;
	#endif

	#ifdef PIMUNU 					// piT coefficients
		precision eta_T;
		precision delta_pipi;
		precision tau_pipi;
		precision lambda_pipi;
		precision lambda_Wupi;
		precision lambda_WTpi;
	#endif

		precision t_020;			// hypergeometric functions
		precision t_001;

		precision t_240;
		precision t_221;
		precision t_202;

		precision t_441;
		precision t_421;
		precision t_402;
		precision t_422;
		precision t_403;

		aniso_transport_coefficients_nonconformal();
		~aniso_transport_coefficients_nonconformal();

		void compute_hypergeometric_functions_n_equals_0(precision z);
		void compute_hypergeometric_functions_n_equals_2(precision z);
		void compute_hypergeometric_functions_n_equals_4(precision z);

		void compute_transport_coefficients(precision e, precision p, precision pl, precision pt, precision b, precision beq, precision lambda, precision aT, precision aL, precision mbar, precision mass, precision mdmde);
};


#endif



