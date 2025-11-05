from pathlib import Path
import torch
import torch.nn.functional as F
from models.multimodal_graph_encoder import MultimodalGraphEncoder
from config import CLIP_MODEL_NAME, FUSED_DIM, PROJ_DIM


def log(*args, **kwargs):
    print(*args, **kwargs)

    with open('./training_log.txt', 'a') as f:
        print(*args, **kwargs, file=f)


def info_nce_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates the InfoNCE loss for a batch of logits.
    Assumes logits are (N, N) and labels are just torch.arange(N).
    """
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_j) / 2.0


def load_pretrained_encoder(
    checkpoint_path: str,
    device: str = "cuda"
) -> MultimodalGraphEncoder:
    """
    Initializes the MultimodalGraphEncoder, loads the pretrained 
    state_dict, and returns it in evaluation mode.

    Args:
        checkpoint_path (str): Path to the .pt file.
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        MultimodalGraphEncoder: The loaded, pretrained model.
    """
    log(f"Loading pretrained encoder from {checkpoint_path}...")

    model = MultimodalGraphEncoder(
        clip_name=CLIP_MODEL_NAME,
        fused_dim=FUSED_DIM,
        proj_dim=PROJ_DIM
    )

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please run 'pretrain_end_to_end.py' first."
        )
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except RuntimeError as e:
        log(f"Error loading state_dict: {e}")
        log("This might happen if the model architecture in util.py "
            "does not match the one used during training.")
        raise e

    model.to(device)
    model.eval()

    log("Pretrained encoder loaded successfully and set to eval mode.")
    return model


# 30 Meta-Relations Grouping
# A dictionary mapping each original relation ID to its new meta-relation.
relation_to_meta = {
    0: "ONTOLOGY_METADATA",  # rdf-schema#seeAlso
    1: "GEO_POLITICAL",  # country
    2: "CULTURE_LANGUAGE",  # language
    3: "GEO_POLITICAL",  # capital
    4: "GEO_POLITICAL",  # largestCity
    5: "LOC_EVENT",  # deathPlace
    6: "LOC_EVENT",  # restingPlace
    7: "POLITICS_GROUP",  # party
    8: "POLITICS_SUCCESSION",  # successor
    9: "GEO_POLITICAL",  # region
    10: "POLITICS_SUCCESSION",  # predecessor
    11: "MEDIA_PERFORMER",  # presenter
    12: "LOC_EVENT",  # birthPlace
    13: "PERSON_ACHIEVEMENT",  # mainInterest
    14: "PERSON_ASSOCIATES",  # influencedBy
    15: "PERSON_BIO",  # citizenship
    16: "ONTOLOGY_METADATA",  # subject
    17: "PERSON_EDUCATION",  # almaMater
    18: "PERSON_ACHIEVEMENT",  # award
    19: "POLITICS_GROUP",  # governmentType
    20: "CULTURE_LANGUAGE",  # officialLanguage
    21: "GEO_OTHER",  # timeZone
    22: "CULTURE_SOCIETY",  # currency
    23: "CULTURE_SOCIETY",  # ethnicGroup
    24: "POLITICS_ROLE",  # leader
    25: "PERSON_EDUCATION",  # education
    26: "GEO_OTHER",  # era
    27: "PERSON_ASSOCIATES",  # influenced
    28: "LOC_GENERAL",  # residence
    29: "PERSON_ACHIEVEMENT",  # field
    30: "LOC_GENERAL",  # place
    31: "LOC_GENERAL",  # territory
    32: "MILITARY_PERSON",  # commander
    33: "ONTOLOGY_METADATA",  # type
    34: "GEO_HYDROLOGY",  # outflow
    35: "PERSON_CAREER",  # occupation
    36: "CULTURE_SOCIETY",  # religion
    37: "POLITICS_GROUP",  # leaderParty
    38: "POLITICS_GROUP",  # governingBody
    39: "ONTOLOGY_METADATA",  # isPartOf
    40: "PERSON_EDUCATION",  # training
    41: "CULTURE_SOCIETY",  # movement
    42: "LOC_GENERAL",  # locationCity
    43: "ORG_PRODUCT",  # industry
    44: "ORG_PRODUCT",  # product
    45: "CULTURE_SOCIETY",  # anthem
    46: "CULTURE_SOCIETY",  # philosophicalSchool
    47: "MEDIA_WORK",  # genre
    48: "PERSON_FAMILY",  # parent
    49: "LOC_GENERAL",  # location
    50: "ORG_STRUCTURE",  # parentCompany
    51: "LOC_STRUCTURE",  # headquarter
    52: "GEO_POLITICAL",  # federalState
    53: "POLITICS_GROUP",  # otherParty
    54: "PERSON_CAREER",  # profession
    55: "MILITARY_PERSON",  # militaryRank
    56: "MILITARY_UNIT_EVENT",  # battle
    57: "POLITICS_ROLE",  # vicePresident
    58: "MILITARY_UNIT_EVENT",  # militaryBranch
    59: "POLITICS_ROLE",  # president
    60: "PERSON_ACHIEVEMENT",  # notableIdea
    61: "PERSON_FAMILY",  # spouse
    62: "PERSON_FAMILY",  # child
    63: "PERSON_FAMILY",  # relative
    64: "ONTOLOGY_METADATA",  # owl#differentFrom
    65: "GEO_POLITICAL",  # city
    66: "GEO_POLITICAL",  # state
    67: "SPORTS_STRUCTURE",  # athletics
    68: "ORG_STRUCTURE",  # affiliation
    69: "CULTURE_LANGUAGE",  # spokenIn
    70: "CULTURE_LANGUAGE",  # languageFamily
    71: "LOC_EVENT",  # foundationPlace
    72: "MEDIA_CREATOR",  # creator
    73: "MEDIA_PERFORMER",  # starring
    74: "MEDIA_CREATOR",  # executiveProducer
    75: "ORG_STRUCTURE",  # company
    76: "MEDIA_INDUSTRY",  # distributor
    77: "MEDIA_INDUSTRY",  # channel
    78: "MEDIA_WORK",  # format
    79: "ONTOLOGY_METADATA",  # related
    80: "GEO_POLITICAL",  # populationPlace
    81: "LOC_EVENT",  # hometown
    82: "SPORTS_STRUCTURE",  # league
    83: "SPORTS_STRUCTURE",  # position
    84: "POLITICS_ROLE",  # leaderName
    85: "LOC_STRUCTURE",  # hubAirport
    86: "ORG_STRUCTURE",  # alliance
    87: "MEDIA_WORK",  # instrument
    88: "POLITICS_GROUP",  # ideology
    89: "MEDIA_INDUSTRY",  # recordLabel
    90: "PERSON_ASSOCIATES",  # associatedBand
    91: "PERSON_ASSOCIATES",  # associatedMusicalArtist
    92: "PERSON_BIO",  # nationality
    93: "MEDIA_INDUSTRY",  # broadcastArea
    94: "PERSON_CAREER",  # employer
    95: "MEDIA_WORK",  # stylisticOrigin
    96: "GEO_OTHER",  # daylightSavingTimeZone
    97: "MEDIA_WORK",  # derivative
    98: "MEDIA_WORK",  # musicSubgenre
    99: "MEDIA_WORK",  # musicFusionGenre
    100: "PERSON_ACHIEVEMENT",  # knownFor
    101: "CULTURE_LANGUAGE",  # regionalLanguage
    102: "MEDIA_CREATOR",  # director
    103: "MEDIA_CREATOR",  # writer
    104: "MEDIA_CREATOR",  # musicComposer
    105: "MEDIA_CREATOR",  # cinematography
    106: "MEDIA_PERFORMER",  # formerBandMember
    107: "PERSON_ASSOCIATES",  # associate
    108: "PERSON_BIO",  # deathCause
    109: "ORG_STRUCTURE",  # department
    110: "LOC_STRUCTURE",  # campus
    111: "MEDIA_CREATOR",  # producer
    112: "MEDIA_PERFORMER",  # narrator
    113: "MEDIA_PERFORMER",  # bandMember
    114: "ORG_STRUCTURE",  # owner
    115: "MEDIA_WORK",  # colour
    116: "POLITICS_GROUP",  # mergedIntoParty
    117: "MEDIA_CREATOR",  # editing
    118: "LOC_GENERAL",  # origin
    119: "ONTOLOGY_METADATA",  # part
    120: "ORG_PRODUCT",  # developer
    121: "MEDIA_CREATOR",  # composer
    122: "MEDIA_INDUSTRY",  # network
    123: "MEDIA_WORK",  # previousWork
    124: "PERSON_ASSOCIATES",  # patron
    125: "GEO_POLITICAL",  # twinTown
    126: "GEO_PHYSICAL",  # lowestPlace
    127: "GEO_HYDROLOGY",  # sourceMountain
    128: "GEO_HYDROLOGY",  # sourcePlace
    129: "POLITICS_ROLE",  # monarch
    130: "POLITICS_ROLE",  # primeMinister
    131: "MEDIA_WORK",  # subsequentWork
    132: "ORG_STRUCTURE",  # keyPerson
    133: "ONTOLOGY_METADATA",  # relation
    134: "GEO_POLITICAL",  # stateOfOrigin
    135: "PERSON_ASSOCIATES",  # partner
    136: "PERSON_ACHIEVEMENT",  # notableWork
    137: "SPORTS_STRUCTURE",  # sport
    138: "GEO_PHYSICAL",  # island
    139: "GEO_POLITICAL",  # nearestCity
    140: "GEO_POLITICAL",  # twinCountry
    141: "GEO_ADMINISTRATIVE",  # municipality
    142: "MEDIA_WORK",  # architecturalStyle
    143: "MEDIA_CREATOR",  # author
    144: "GEO_POLITICAL",  # usingCountry
    145: "LOC_GENERAL",  # locationCountry
    146: "ORG_STRUCTURE",  # owningCompany
    147: "MEDIA_INDUSTRY",  # sisterStation
    148: "ORG_STRUCTURE",  # division
    149: "LOC_GENERAL",  # locatedInArea
    150: "GEO_PHYSICAL",  # kingdom
    151: "POLITICS_GROUP",  # politicalPartyInLegislature
    152: "LOC_STRUCTURE",  # ground
    153: "GEO_POLITICAL",  # largestSettlement
    154: "LOC_STRUCTURE",  # significantBuilding
    155: "ORG_PRODUCT",  # license
    156: "GEO_ADMINISTRATIVE",  # countySeat
    157: "MILITARY_UNIT_EVENT",  # isPartOfMilitaryConflict
    158: "LOC_STRUCTURE",  # highschool
    159: "LOC_STRUCTURE",  # college
    160: "SPORTS_PERSON_TEAM",  # draftTeam
    161: "MILITARY_UNIT_EVENT",  # commandStructure
    162: "ORG_STRUCTURE",  # foundedBy
    163: "GEO_PHYSICAL",  # lowestMountain
    164: "POLITICS_ROLE",  # chairman
    165: "ORG_STRUCTURE",  # nationalAffiliation
    166: "GEO_ADMINISTRATIVE",  # canton
    167: "PERSON_BIO",  # ethnicity
    168: "GEO_HYDROLOGY",  # inflow
    169: "GEO_HYDROLOGY",  # leftTributary
    170: "GEO_HYDROLOGY",  # rightTributary
    171: "GEO_HYDROLOGY",  # sourceConfluenceRegion
    172: "GEO_HYDROLOGY",  # sourceRegion
    173: "GEO_HYDROLOGY",  # riverMouth
    174: "GEO_HYDROLOGY",  # mouthMountain
    175: "GEO_HYDROLOGY",  # mouthPlace
    176: "ORG_STRUCTURE",  # institution
    177: "SPORTS_PERSON_TEAM",  # team
    178: "PERSON_BIO",  # gender
    179: "MEDIA_PERFORMER",  # voice
    180: "PERSON_ASSOCIATES",  # academicAdvisor
    181: "GEO_HYDROLOGY",  # sourceConfluenceMountain
    182: "GEO_HYDROLOGY",  # sourceConfluencePlace
    183: "GEO_HYDROLOGY",  # sourceConfluenceState
    184: "ORG_STRUCTURE",  # subsidiary
    185: "ORG_PRODUCT",  # ingredient
    186: "ORG_STRUCTURE",  # sisterCollege
    187: "ORG_PRODUCT",  # manufacturer
    188: "ORG_STRUCTURE",  # founder
    189: "LOC_STRUCTURE",  # targetAirport
    190: "MILITARY_UNIT_EVENT",  # garrison
    191: "GEO_HYDROLOGY",  # sourceCountry
    192: "GEO_HYDROLOGY",  # mouthRegion
    193: "GEO_HYDROLOGY",  # mouthCountry
    194: "MILITARY_UNIT_EVENT",  # aircraftTransport
    195: "ORG_PRODUCT",  # service
    196: "MEDIA_WORK",  # depiction
    197: "POLITICS_GROUP",  # jurisdiction
    198: "ORG_STRUCTURE",  # parentOrganisation
    199: "ORG_STRUCTURE",  # childOrganisation
    200: "GEO_ADMINISTRATIVE",  # ceremonialCounty
    201: "POLITICS_GROUP",  # house
    202: "POLITICS_GROUP",  # politicalPartyOfLeader
    203: "POLITICS_ROLE",  # head
    204: "MILITARY_PERSON",  # secondCommander
    205: "POLITICS_ROLE",  # provost
    206: "MEDIA_INDUSTRY",  # regionServed
    207: "ORG_STRUCTURE",  # board
    208: "MILITARY_UNIT_EVENT",  # equipment
    209: "CULTURE_LANGUAGE",  # languageRegulator
    210: "GEO_ADMINISTRATIVE",  # frazioni
    211: "ONTOLOGY_METADATA",  # class
    212: "POLITICS_ROLE",  # mayor
    213: "GEO_ADMINISTRATIVE",  # district
    214: "GEO_HYDROLOGY",  # source
    215: "GEO_HYDROLOGY",  # river
    216: "GEO_ADMINISTRATIVE",  # county
    217: "MEDIA_CREATOR",  # animator
    218: "MEDIA_WORK",  # series
    219: "POLITICS_ROLE",  # chancellor
    220: "SPORTS_PERSON_TEAM",  # formerTeam
    221: "GEO_ADMINISTRATIVE",  # borough
    222: "MEDIA_INDUSTRY",  # broadcastNetwork
    223: "GEO_ADMINISTRATIVE",  # councilArea
    224: "GEO_ADMINISTRATIVE",  # lieutenancyArea
    225: "MEDIA_INDUSTRY",  # distributingLabel
    226: "MEDIA_INDUSTRY",  # distributingCompany
    227: "GEO_POLITICAL",  # province
    228: "GEO_PHYSICAL",  # archipelago
    229: "SPORTS_PERSON_TEAM",  # debutTeam
    230: "PERSON_ACHIEVEMENT",  # honours
    231: "SPORTS_PERSON_TEAM",  # coach
    232: "CULTURE_SOCIETY",  # veneratedIn
    233: "MEDIA_INDUSTRY",  # publisher
    234: "MEDIA_WORK",  # basedOn
    235: "LOC_GENERAL",  # routeJunction
    236: "LOC_GENERAL",  # routeStart
    237: "LOC_GENERAL",  # routeEnd
    238: "POLITICS_ROLE",  # viceChancellor
    239: "SPORTS_PERSON_TEAM",  # managerClub
    240: "PERSON_BIO",  # person
    241: "MEDIA_WORK",  # openingTheme
    242: "MEDIA_WORK",  # endingTheme
    243: "GEO_ADMINISTRATIVE",  # localAuthority
    244: "SPORTS_STRUCTURE",  # rival
    245: "MEDIA_CREATOR",  # artist
    246: "MEDIA_PERFORMER",  # showJudge
    247: "MEDIA_PERFORMER",  # voiceType
    248: "POLITICS_GROUP",  # youthWing
    249: "GEO_ADMINISTRATIVE",  # metropolitanBorough
    250: "PERSON_EDUCATION",  # grades
    251: "GEO_ADMINISTRATIVE",  # principalArea
    252: "TECH_SPEC",  # cpu
    253: "MILITARY_PERSON",  # notableCommander
    254: "LOC_STRUCTURE",  # stadium
    255: "TECH_SPEC",  # operatingSystem
    256: "MEDIA_INDUSTRY",  # formerBroadcastNetwork
    257: "MEDIA_PERFORMER",  # portrayer
    258: "MEDIA_CREATOR",  # editor
    259: "MEDIA_CREATOR",  # designer
    260: "TECH_SPEC",  # computingPlatform
    261: "LOC_GENERAL",  # map
    262: "LOC_EVENT",  # recordedIn
    263: "MEDIA_WORK",  # programmeFormat
    264: "POLITICS_ROLE",  # governor
    265: "LOC_STRUCTURE",  # majorShrine
    266: "GEO_PHYSICAL",  # species
    267: "MEDIA_WORK",  # openingFilm
    268: "MEDIA_WORK",  # closingFilm
    269: "LOC_EVENT",  # canonizedPlace
    270: "MEDIA_WORK",  # televisionSeries
    271: "SPORTS_STRUCTURE",  # sportGoverningBody
    272: "LOC_EVENT",  # releaseLocation
    273: "MEDIA_WORK",  # literaryGenre
    274: "POLITICS_GROUP",  # splitFromParty
    275: "MEDIA_PERFORMER",  # musicalArtist
    276: "MEDIA_PERFORMER",  # musicalBand
    277: "SPORTS_STRUCTURE",  # promotion
    278: "ONTOLOGY_METADATA"  # 22-rdf-syntax-ns#type
}
